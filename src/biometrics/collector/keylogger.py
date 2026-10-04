from biometrics.inference.xgboost_inference import Inference
from collections import Counter
from pynput.keyboard import Key, Listener, KeyCode
from biometrics.collector.logging_utils import save_csv, get_keyboard_language
from sys import exit
import os
import time
import datetime

timeout = 10  #Seconds until we decide that we are not at the keyboard


class _Listener(Listener):
    """
    pynput's uinput backend stops by closing the evdev device. When stop() is
    called from inside a key callback (ESC), read_loop() then comes back
    around and select()s on the closed fd (-1): ValueError/OSError, printed
    as an unhandled thread exception. pynput marks the listener as not
    running *before* closing the device, so an error after that point is just
    the expected end of the loop.
    """

    def _run(self):
        try:
            super()._run()
        except (ValueError, OSError):
            if self.running:
                raise


def _uinput_keyboard_device_paths():
    """
    pynput's uinput backend auto-picks whichever /dev/input/eventX reports
    the most EV_KEY codes. On a machine with a gaming mouse whose extra
    buttons register as a synthetic "keyboard" input device, that heuristic
    can pick the mouse's fake keyboard node over the real keyboard, so
    typing produces no events at all. Only relevant when
    PYNPUT_BACKEND(_KEYBOARD)=uinput is active; returns None on any other
    backend or if evdev/the real device can't be resolved, so callers can
    fall back to pynput's own default device selection.
    """
    if "uinput" not in (os.environ.get("PYNPUT_BACKEND_KEYBOARD")
                         or os.environ.get("PYNPUT_BACKEND") or ""):
        return None

    try:
        import evdev
    except ImportError:
        return None

    candidates = []
    for path in evdev.list_devices():
        try:
            dev = evdev.InputDevice(path)
        except OSError:
            continue
        name = dev.name.lower()
        if "mouse" in name or "gamepad" in name or "pad" in name:
            continue
        key_count = len(dev.capabilities().get(evdev.ecodes.EV_KEY, []))
        if key_count > 50:
            candidates.append((key_count, path))

    if not candidates:
        return None
    candidates.sort(reverse=True)
    return [candidates[0][1]]

class KeyLogger:

    def __init__(self, label, file_path, threshold=50, model_paths=None, on_prediction=None):
        self.file_path = file_path #Where to save the file
        self.label = label # Who does the logging
        self.pressed_keys = set() #Aux set to hold the pressed_keys
        self.data = [] # Variable to hold the logging data temporarily
        self.total_start_time = 0 #Variable to hold the start time of the logging
        self.key_counter = 0 # Total pressed keys
        self.key_combinations = Counter() #Count the frequences of combination
        self.threshold = threshold # How frequently we should save
        self.error_counter = 0 # Total errors
        self.sequence_length = threshold # Sequence threshold where we begin a new file
        self.inference_mode = False #Inference or not
        self.on_prediction = on_prediction # Optional callback(label, confidence, feature_row, keystroke_window), e.g. the protection agent
        self._esc_pressed = False # Set by on_press, read by start_logging()'s wrapper to stop the listener

        #Inference Mode
        self.model_paths = model_paths
        if model_paths:
            self.inference_mode = True
            self.inference_model = Inference(model_paths, 0.9, threshold)


    def reload_model(self):
        """Switch to a freshly (re)trained model without stopping the listener."""
        self.inference_model = Inference(self.model_paths, 0.9, self.threshold)
        print(f"\n{datetime.datetime.now()} - New model loaded.")


    def new_state(self):
        '''
        Function to reset the state when we start a new file (Reset counters, time etc.)
        '''
        self.key_counter = 0 # Total pressed keys
        self.total_start_time = time.time() #Start of the logging
        self.pressed_keys = set() #Aux set to hold the pressed_keys
        self.data = [] # Variable to hold the logging data temporarily
        self.key_combinations = Counter() #Count the frequences of combination
        self.error_counter = 0


    def calculate_kps(self, elapsed_time):
        """
        Function to calculate the current average key per second
        """

        return self.key_counter / elapsed_time if elapsed_time > 0 else 0


    def check_threshold(self):
        '''
        Function to check if we reached the threshold then save, in inference we predict
        '''
        if self.inference_mode:
            print(f"\r  captured {self.key_counter}/{self.threshold} keystrokes...", end="", flush=True)

        if self.key_counter % self.threshold == 0:
            if self.inference_mode:
                result = self.inference_model.predict(self.data, return_feature_row=True)
                if result is not None and self.on_prediction is not None:
                    predicted_label, predicted_confidence, feature_row = result
                    self.on_prediction(predicted_label, predicted_confidence,
                                        feature_row=feature_row, keystroke_window=list(self.data))
            else:
                #save the file as CSV
                self.save_logging()
            self.new_state()


    def reset_data(self):
        '''
        Function to reset the data, after keyboard inactivity or reaching threshold
        '''
        self.total_start_time = time.time()
        self.data[-1]['seek_time'] = 0
        self.data[-1]['key_per_second'] = 0
        self.data[-1]['start_time'] = 0
        self.data[-1]['key_per_second'] = 0
        self.key_counter = 1


    def on_press_append_data(self, key, st, st_timestamp):
        """
        Function to prepare the data for save, we need this function as
        we have 2 different function for press and release actions
        :param key: Key pressed
        :param st: Start time of the key pressed
        :param st_timestamp The start timestamp
        """

        #Current time from the start
        st = st - self.total_start_time

        #New entry
        self.data.append({"key" : str(key),
                        "start_time" : st,
                        "key_per_second" : self.calculate_kps(st),
                        "combination" : None,
                        "error" : False,
                        "start_time_timestamp" : st_timestamp})

        #If this is not the first entry then we can calculate the seek time, else this was the first key press
        #If the previous key pressed is a combination, then we take the start time of the previous one as seek time
        if len(self.data) > 1:
            if self.data[-2].get('end_time', None) is not None:
                if(st - self.data[-2]['end_time'] > timeout):
                    self.reset_data()
                else:
                    self.data[-1]['seek_time'] = st - self.data[-2]['end_time']
            else:
                if(st - self.data[-2]['start_time'] > timeout):
                    self.reset_data()
                else:
                    self.data[-1]['seek_time'] = st - self.data[-2]['start_time']
        else:
            self.data[-1]['seek_time'] = 0
            self.data[-1]['key_per_second'] = 0

        #Add error rate
        self.data[-1]["error_rate"] = self.error_counter / self.key_counter

        #Add accuracy
        self.data[-1]["accuracy"] = (self.key_counter - self.error_counter) / self.key_counter

        #If there are multiple keys pressed, then it's a combination, we save this combination, and increase its counter
        if len(self.pressed_keys) > 1:
            s = ""
            for c in self.pressed_keys:
                s += c + ","
            self.data[-1]['combination'] = s[:-1]
            self.key_combinations[s[:-1]] += 1

        #Check if the key that was pressed it was an error and mark that key as error
        if key == Key.backspace:
            loc = -1
            while abs(loc) < len(self.data)-1 and str(self.data[loc]['key']) == str(Key.backspace):
                loc -= 1

            while abs(loc) < len(self.data)-1 and str(self.data[loc]['error']) == str(True):
                loc -= 1

            self.error_counter += 1
            self.data[loc]['error'] = True
            self.data[loc]["error_rate"] = self.error_counter / self.key_counter
            self.data[loc]["accuracy"] = (self.key_counter - self.error_counter) / self.key_counter


    def on_press(self, key):
        if len(self.data) == 0:
            self.total_start_time = time.time() #Start of the logging

        #Get the time of the press
        start_time = time.time()

        #Cancel the logging: start_logging()'s wrapper sees the flag and stops
        #the listener (raising/returning False here isn't handled by pynput's
        #uinput backend -- see _Listener).
        if key == Key.esc:
            self._esc_pressed = True
            return

        #Increase the simultaneously pressed keys list
        if str(key).upper() not in self.pressed_keys:
            self.pressed_keys.add(str(key).upper())

        # Increase the total key count for final display
        self.key_counter += 1

        #Save the pressed key's statistics
        self.on_press_append_data(key, start_time, start_time)


    def on_release_append_data(self, key, et, ts, lang):
        index = -1

        #We search for the last not released key
        try:
            while key.upper() != self.data[index]['key'].upper():
                index -= 1
        except IndexError:
            return

        #If there are more than one same key next to each other, we find the first one not updated
        try:
            while 'end_time' not in self.data[index]:
                if self.data[index-1]['key'].upper() == key.upper() and 'end_time' not in self.data[index-1]:
                    index -= 1
                else:
                    break
        except IndexError:
            pass

        #Get the time since start of the logging
        et = et - self.total_start_time

        #Insert end_time and calculate hold_time
        self.data[index]['end_time'] = et
        self.data[index]['hold_time'] = et - self.data[index]['start_time']
        self.data[index]['end_time_timestamp'] = ts

        #We left the keyboard
        if et - self.data[index]['start_time'] < 0:
            self.data[index]['hold_time'] = 0

        self.data[index]['language'] = lang


    def on_release(self, key):
        #Get the time of the release
        end_time = time.time()

        keyboard_language = get_keyboard_language() # Get the current keyboard language used

        #Bugfix for hungarian keyboard layout
        if keyboard_language == "Hungarian":
            if str(key) == "<90>":
                key = KeyCode.from_char("z")
            elif str(key) == "'j'" and "'Í'" in self.pressed_keys:
                key = KeyCode.from_char("í")

        #Pop the key we process from the pressed keys
        self.pressed_keys.discard(str(key).upper())

        #Update the other part of the data
        self.on_release_append_data(str(key), end_time, end_time, keyboard_language)

        self.check_threshold()


    def start_logging(self):
        """
        Initializes the text, the parameters, and starts the logging
        :return:
        """
        if self.inference_mode:
            print(f"{datetime.datetime.now()} - Inference started!")
        else:
            print(f"{datetime.datetime.now()} - Logging started!")


        #Start the listener
        self._esc_pressed = False

        def _on_press(key):
            self.on_press(key)
            if self._esc_pressed:
                listener.stop()

        listener_kwargs = {"on_press": _on_press, "on_release": self.on_release}
        device_paths = _uinput_keyboard_device_paths()
        if device_paths:
            print(f"Using keyboard device: {device_paths[0]}")
            listener_kwargs["uinput_device_paths"] = device_paths

        listener = _Listener(**listener_kwargs)
        with listener:
            listener.join()

        self.end_logging()

    def end_logging(self):
        """
        Function to safely end the logging
        :return:
        """
        if not self.inference_mode:
            self.save_logging()

            print(f"{datetime.datetime.now()} - Logging finished!")

    def save_logging(self):
        """
        Function to save the collected logger data
        :return:
        """

        try:
            save_csv(self.file_path, self.label, self.data)
        except IndexError:
            return

        print(f"{datetime.datetime.now()} - Logger state saved!")
