import matplotlib.pyplot as plt
import numpy as np
import json
from matplotlib.widgets import Slider
from scipy.io import wavfile
from numba import njit
from tesfuncs.funcs import *
from plot_bifurcations import plot_bifs
from signal_filters import low_pass_filter
from integrators import RK4
from color_gradient import color_gradient

def main():
    gestures_path = "json_files/gestures.json"
    imitation_path = "json_files/imitation_params.json"
    NUMBER_OF_TERMS = 28

    #  colors = ['#005fff', '#9259d2', '#b65bb5', '#cb629f', '#d86c8e', '#e17881',
              #  '#e58676', '#e6956d', '#e4a465', '#deb55e', '#d3c658', '#c1d751',
              #  '#a1ea49', '#51ff3d', '#22ff00']


    x0 = np.array([0.0705, 0.15942])
    times = np.arange(0, 3.4, 0.001)
    wc_terms = []
    colors = color_gradient("#0023ff", "#fa9b00", NUMBER_OF_TERMS)

    for i in range(NUMBER_OF_TERMS):
        wc_terms.append(KickedWilsonCowan(x0, times, -2, -6, 200, -0.0, 1, 1, colors[i]))

    additive_wc_model = AdditiveWilsonCowan(*wc_terms)
    interactive_figure = InteractiveFigure(additive_wc_model,
                                           gestures_path, imitation_path,
                                           starting_index=1159)


    slider_settings = [("target", 0, NUMBER_OF_TERMS-1, 0, "master"),
                       ("kick_y", -1, 0, -0, "kick"),
                       ("kicktime_y", 0.8, 3, 1, "kicktime"),
                       ("scale_y", 0, 3, 1, "scale"),
                       ("mu", 50, 250, 100, "mu"),
                       ("bottom shift", 0, 1, 0, "bottom")]

    for setting in reversed(slider_settings):
        interactive_figure.add_slider(*setting)
    plt.show()

@njit
def wilsoncowan(z, t, rox, roy, mu):
    x, y = z
    x_dot = mu*(-x + 1/(1+np.exp(-(rox + 10*x - 10*y))))
    y_dot = mu*(-y + 1/(1+np.exp(-(roy + 10*x + 10*y))))
    return np.array([x_dot, y_dot])

class KickedWilsonCowan:
    def __init__(self, x0: np.ndarray, times: np.ndarray, rox: float,
                 roy: float, mu: float, kick: float, kicktime: float,
                 scale: float, color: str):
        self.x0 = x0
        self.times = times
        self.params = {"rox": rox,
                       "roy": roy,
                       "mu": mu,
                       "kick": kick,
                       "kicktime": kicktime}
        self.scale = scale
        self.color = color
        self.integrate()

    def integrate(self):
        times = self.times
        prekick_times = times[times < self.params["kicktime"]]
        postkick_times = times[times >= self.params["kicktime"]]
        prekick_sol = RK4(wilsoncowan, self.x0, prekick_times,
                    self.params["rox"], self.params["roy"], self.params["mu"])
        vector_kick = np.array([0, self.params["kick"]])
        postkick_sol = RK4(wilsoncowan, prekick_sol[-1] + vector_kick,
                        postkick_times, self.params["rox"], self.params["roy"],
                        self.params["mu"])
        self.signal = np.concatenate([prekick_sol, postkick_sol])

    def update_param(self, param: str, new_value: float):
        self.params[param] = new_value
        self.integrate()

    def get_params(self, term_i: int):
        params_i_dict = {f"{key}{term_i}": value for key,value in self.params.items()}
        return {**params_i_dict, **{f"scale{term_i}": self.scale}}

class AdditiveWilsonCowan:
    def __init__(self, *additive_terms):
        self.additive_terms = {i: term for i, term in enumerate(additive_terms)}

    def add_term(self, *additive_terms):
        self.additive_terms = {**self.additive_terms, **{i+len(self.additive_terms.keys()): term for i, term in enumerate(additive_terms)}}


class InteractiveFigure:
    def __init__(self, linked_system, gesture_path, imitation_path, starting_index=0):
        self.linked_system = linked_system
        self.total_sliders = 0
        self.sliders_dict = {}
        self.roxroy_plots = {}
        self.solution_plots = {}
        self.bottom_shift = 0
        self.gesture_index = starting_index
        self.gesture_path = gesture_path
        self.imitation_path = imitation_path
        self.plot_bifs_and_layout()
        self.plot_initial_conditions()
        self.plot_sum()

        self.cid_pick_rho_onpress = self.fig.canvas.mpl_connect('button_press_event', self.pick_rox_roy)
        self.cid_pick_rho_onmove = self.fig.canvas.mpl_connect("motion_notify_event", self.pick_rox_roy)
        self.cid_key_press = self.fig.canvas.mpl_connect("key_press_event", self.key_press)
        self.cid_change = self.fig.canvas.mpl_connect("button_press_event", self.change_gesture)

    def plot_initial_conditions(self):
        for term_key, term_value in self.linked_system.additive_terms.items():
            roxroy_plot, = self.ax_left.plot([term_value.params["rox"]],
                [term_value.params["roy"]], color=term_value.color, marker="o")
            self.roxroy_plots[term_key] = roxroy_plot
            solution_plot, = self.ax_right.plot(term_value.times,
                term_value.signal[:,0], color=term_value.color)
            self.solution_plots[term_key] = solution_plot

    def plot_bifs_and_layout(self):
        self.fig, (self.ax_left, self.ax_right) = plt.subplots(1, 2, figsize=(16, 9))
        self.ax_right.set_ylabel("x(t)")
        self.ax_right.set_ylim(-0.05, 1.05)
        self.ax_right.set_xlim(0.95, max(self.linked_system.additive_terms[0].times))
        self.gesture_plot, = self.ax_right.plot([],[], color="#999", alpha=0.6)
        plot_bifs(4000, self.ax_left, 10, 10, 10, -10)

    def plot_sum(self):
        term_value = self.linked_system.additive_terms[0]
        sum_curves = sum([sol.get_ydata() for sol in self.solution_plots.values()])
        self.sum_series, = self.ax_right.plot(term_value.times,
                            self.bottom_shift + sum_curves, "r--")

    def update_plot(self):
        sum_curves = sum([sol.get_ydata() for sol in self.solution_plots.values()])
        self.sum_series.set_ydata(self.bottom_shift + sum_curves)
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def add_slider(self, label: str, minval: float, maxval: float,
                   valinit: float, param: str):
        self.fig.subplots_adjust(bottom=(len(self.sliders_dict.keys())+1)*0.03+0.1)
        ax = self.fig.add_axes([0.15, len(self.sliders_dict.keys())*0.03, 0.75, 0.03])
        slider = TargetSlider(self.linked_system, self, param, ax, label, minval, maxval, valinit=valinit)
        if param == "master":
            slider.valstep = np.arange(minval, maxval+1, 1, dtype=int)
            slider.set_as_master(self.sliders_dict.values())
            self.slider_master = slider
        else:
            slider.set_as_parameter_of(0)
            self.sliders_dict[label] = slider

    def pick_rox_roy(self, event):
        zoom_status = self.ax_left.get_navigate_mode() == "ZOOM"
        in_left_ax = event.inaxes == self.ax_left
        if (event.button == 1) and in_left_ax and not zoom_status:
            rox, roy = event.xdata, event.ydata
            target_term = self.linked_system.additive_terms[self.slider_master.val]
            target_term.params["rox"] = rox
            target_term.params["roy"] = roy
            self.roxroy_plots[self.slider_master.val].set_data([rox], [roy])
            target_term.integrate()
            self.solution_plots[self.slider_master.val].set_ydata(
                    target_term.scale*target_term.signal[:,0])
            self.update_plot()

    def change_gesture(self, event):
        zoom_status = self.ax_left.get_navigate_mode() == "ZOOM"
        with open(self.gesture_path) as gp:
            gesture_data = json.load(gp)

        if str(event.inaxes) == str(self.ax_right) and not zoom_status:
            if event.button == 1:
                self.gesture_index -= 1
            elif event.button == 3:
                self.gesture_index += 1
            self.gesture_index %= len(gesture_data.keys())
            index_dict = gesture_data[str(self.gesture_index)]
            with open(f"{index_dict['dir_path']}/{index_dict['file']}", "rb") as wav:
                fs, p_wav = wavfile.read(wav)

            if "channel" in index_dict.keys():
                p_wav = p_wav[:, index_dict["channel"]]
            p_wav = p_wav[int(index_dict['t0']*fs):int(index_dict['tf']*fs)]
            p_wav = p_wav.astype("float")
            p_wav = p_wav-min(p_wav)
            p_wav = p_wav/max(p_wav)
            times = np.linspace(1, 1+index_dict['tf']-index_dict['t0'], len(p_wav))
            p_wav = low_pass_filter(p_wav, fs)
            self.gesture_plot.set_data(times, p_wav)
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()


    def key_press(self, event):
        if event.key == "enter":
            self.save_parameters()
        if event.key == "tab":
            self.next_additive_term()

    def save_parameters(self):
        with open(self.gesture_path) as gp:
            gesture_data = json.load(gp)
        with open(self.imitation_path) as ip:
            imitation_data = json.load(ip)
        i_dict = gesture_data[str(self.gesture_index)]
        imitation_index = len(imitation_data)
        data_dict = {"data_index": self.gesture_index,
                     "dir_path": i_dict["dir_path"],
                     "file": i_dict["file"],
                     "t0": i_dict["t0"],
                     "tf": i_dict["tf"]}
        for i in range(len(self.linked_system.additive_terms.keys())):
            i_term_dict = self.linked_system.additive_terms[i].get_params(i+1)
            data_dict = {**data_dict, **i_term_dict}
        data_dict = {**data_dict, **{"bottom_shift": self.bottom_shift}}

        imitation_data[imitation_index] = data_dict
        with open(self.imitation_path, "w") as f:
            json.dump(imitation_data, f, ensure_ascii=False)




class TargetSlider(Slider):
    def __init__(self, target_model, linked_figure, param: str, *args, target_index = 0, **kwargs):
        Slider.__init__(self, *args, **kwargs)
        self.target_model = target_model
        self.linked_figure = linked_figure
        self.target = target_index

        self.param = param
        self.label.set_size(16)
        self.valtext.set_size(16)

        self.cid_list = []

    def change_target(self, target):
        self.target = target

    def set_as_master(self, child_sliders):
        def action(val):
            for slider in child_sliders:
                slider.change_target(val)
                slider.set_as_parameter_of(int(val))
        self.on_changed(action)

    def set_as_parameter_of(self, target_index):
        for cid in self.cid_list:
            self.disconnect(cid)
        target_term = self.target_model.additive_terms[target_index]

        def action(val):
            if self.param == "scale":
                target_term.scale = val
                self.linked_figure.solution_plots[target_index].set_ydata(
                                    target_term.scale*target_term.signal[:,0])
            elif self.param == "bottom":
                self.linked_figure.bottom_shift = val
            else:
                target_term.update_param(self.param, val)
                self.linked_figure.solution_plots[target_index].set_ydata(
                                    target_term.scale*target_term.signal[:,0])

            self.linked_figure.update_plot()

        cid = self.on_changed(action)
        self.cid_list.append(cid)

if __name__ == "__main__":
    main()
