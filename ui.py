import gtk
import gobject
import hashlib
import af
import numpy as np

from matplotlib.figure import Figure
from matplotlib.backends.backend_gtkagg import FigureCanvasGTKAgg as FigureCanvas


def make_grey(arr):
    a = arr.astype(np.ubyte)
    new = np.dstack((a, a))
    return np.dstack((new, a))


class FocusCanvas(gtk.DrawingArea):

    __gsignals__ = {
        'expose-event': 'override',
        'region-updated': (gobject.SIGNAL_RUN_LAST, gobject.TYPE_NONE,
            (gobject.TYPE_FLOAT, gobject.TYPE_FLOAT, gobject.TYPE_FLOAT,
                gobject.TYPE_FLOAT, ))}

    def __init__(self):
        super(FocusCanvas, self).__init__()
        self.region_start = [0, 0]
        self.region_width = 10.0
        self.region_height = 10.0
        self.current_mouse_pos = (0, 0)
        self.button_pressed = False
        self.current_image = None
        self.image_cache = {}
        self.add_events(gtk.gdk.POINTER_MOTION_MASK |
                        gtk.gdk.BUTTON_PRESS_MASK |
                        gtk.gdk.BUTTON_RELEASE_MASK)

        self.connect('button-press-event', self._on_button_press)
        self.connect('button-release-event', self._on_button_release)
        self.connect('motion-notify-event', self._on_motion)

    def do_expose_event(self, event):
        cr = self.window.cairo_create()
        cr.rectangle(event.area.x, event.area.y, event.area.width,
                event.area.height)
        cr.clip()
        self._draw(cr, *self.window.get_size())

    def update_image(self, image):
        h = hashlib.sha1(image.view(np.uint8))
        if h in self.image_cache:
            self.current_image = self.image_cache[h]
        else:
            self.current_image = gtk.gdk.pixbuf_new_from_array(
                    make_grey(image), gtk.gdk.COLORSPACE_RGB, 8)
            self.image_cache[h] = self.current_image
        self.queue_draw()

    def _on_button_press(self, widget, data):
        self.region_start[0] = self.current_mouse_pos[0]
        self.region_start[1] = self.current_mouse_pos[1]
        self.button_pressed = True

    def _on_button_release(self, widget, data):
        self.button_pressed = False
        rx, ry = self.region_start[0], self.region_start[1]
        width, height = self.region_width, self.region_height
        if width < 0.0:
            rx += width
            width = abs(width)
        if height < 0.0:
            ry += height
            height = abs(height)
        self.emit('region-updated', rx, ry, width, height)

    def _on_motion(self, widget, data):
        mp = self.current_mouse_pos = data.get_coords()
        if self.button_pressed:
            self.region_width = mp[0] - self.region_start[0]
            self.region_height = mp[1] - self.region_start[1]
        self.queue_draw()

    def _draw(self, cr, width, height):
        cr.set_source_rgb(1.0, 1.0, 1.0)
        cr.rectangle(0, 0, width, height)
        cr.fill()

        if self.current_image:
            cr.set_source_pixbuf(self.current_image, 1, 1)
            cr.paint()

        cr.set_source_rgb(0.204, 0.541, 0.741)
        cr.set_line_width(2.0)
        r = self.region_start
        cr.rectangle(r[0] + 0.5, r[1] + 0.5, self.region_width,
                self.region_height)
        cr.stroke()


class FigureWidget(gtk.ScrolledWindow):
    def __init__(self, figure):
        super(FigureWidget, self).__init__()
        self.set_border_width(10)
        self.set_policy(hscrollbar_policy=gtk.POLICY_AUTOMATIC,
            vscrollbar_policy=gtk.POLICY_ALWAYS)
        self.canvas = FigureCanvas(figure)
        self.canvas.set_size_request(500, 100)
        self.add_with_viewport(self.canvas)


class Application(object):
    def __init__(self):
        self.stack = af.load_image_stack("data/*.jpg")

        win = gtk.Window()
        win.connect("destroy", lambda x: gtk.main_quit())
        win.set_default_size(960, 540)
        win.set_title("Autofocus Test")

        vpane = gtk.VPaned()
        win.add(vpane)

        self.canvas = FocusCanvas()
        self.canvas.connect('region-updated', self.on_region_update)
        self.canvas.update_image(self.stack[:,:,0])
        vpane.add(self.canvas)

        self.figure = Figure(figsize=(5, 1), dpi=100)
        self.ax = self.figure.add_subplot(111)
        self.figure_widget = FigureWidget(self.figure)
        vpane.add(self.figure_widget)

        win.show_all()

    def on_region_update(self, widget, x, y, width, height):
        fp = af.FocusPoint(x, y, width, height)

        J_sobel = af.optimize(self.stack, fp, af.cost_sobel)
        J_grad = af.optimize(self.stack, fp, af.cost_gradient)
        J_stddev = af.optimize(self.stack, fp, af.cost_stddev)

        self.ax.clear()
        self.ax.plot(J_sobel / np.max(J_sobel), '.-', label='Sobel')
        self.ax.plot(J_grad / np.max(J_grad), '.-', label='Gradient')
        self.ax.plot(J_stddev / np.max(J_stddev), '.-', label='Stddev')
        self.ax.legend(loc=3)
        self.figure.canvas.draw()

        opt = np.argmax(J_sobel)
        self.canvas.update_image(self.stack[:,:,opt])


if __name__ == '__main__':
    app = Application()
    gtk.main()
