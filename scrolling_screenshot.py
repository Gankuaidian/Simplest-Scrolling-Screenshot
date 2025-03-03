import time
import argparse
import pyautogui
from PIL import Image, ImageDraw, ImageTk
import keyboard
import os
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, colorchooser
from pynput import mouse
import numpy as np
import cv2

class ScrollingScreenshotTool:
    def __init__(self, output_dir="screenshots"):
        self.output_dir = output_dir
        self.region = None  
        self.final_image = None
        self.last_screenshot = None
        self.current_y_offset = 0
        self.total_height = 0
        self.is_capturing = False
        self.scroll_buffer = []  
        self.draw_color = (255, 0, 0)  
        self.draw_lines = []  
        self.is_drawing = False  
        self.current_line = []  
        self.drawing_overlay = None  
        self.scale = 1.0  
        self.zoom_factor = 1.0  

        self.listener = mouse.Listener(on_scroll=self.on_scroll)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
    
    def on_scroll(self, x, y, dx, dy):
        if self.is_capturing and self.region:
            time.sleep(0.1)
            self.detect_and_capture_new_content()
    
    def detect_and_capture_new_content(self):
        current_screenshot = pyautogui.screenshot(region=self.region)
        if self.last_screenshot is None:
            self.last_screenshot = current_screenshot
            self.scroll_buffer.append(current_screenshot)
            return
        
        prev_img = np.array(self.last_screenshot)
        curr_img = np.array(current_screenshot)
        prev_gray = cv2.cvtColor(prev_img, cv2.COLOR_RGB2GRAY)
        curr_gray = cv2.cvtColor(curr_img, cv2.COLOR_RGB2GRAY)
        
        scroll_amount = self.calculate_scroll_amount(prev_gray, curr_gray)
        
        if scroll_amount > 10:
            self.scroll_buffer.append(current_screenshot)
            print(f"Detected scroll of {scroll_amount} pixels")
            
            self.process_scroll_buffer()
            
            self.last_screenshot = current_screenshot
    
    def calculate_scroll_amount(self, prev_img, curr_img):
        max_scroll = min(prev_img.shape[0], curr_img.shape[0]) // 2
        best_match = 0
        scroll_amount = 0
        
        for shift in range(10, max_scroll, 10):
            prev_section = prev_img[-shift:, :]
            curr_section = curr_img[:shift, :]
            if prev_section.shape == curr_section.shape:
                match_score = np.sum(prev_section == curr_section) / (prev_section.size)
                if match_score > best_match:
                    best_match = match_score
                    scroll_amount = shift
        
        if scroll_amount > 0:
            refined_start = max(10, scroll_amount - 10)
            refined_end = min(max_scroll, scroll_amount + 10)
            for shift in range(refined_start, refined_end):
                prev_section = prev_img[-shift:, :]
                curr_section = curr_img[:shift, :]
                if prev_section.shape == curr_section.shape:
                    match_score = np.sum(prev_section == curr_section) / (prev_section.size)
                    if match_score > best_match:
                        best_match = match_score
                        scroll_amount = shift
        
        if best_match < 0.6:
            template_height = min(100, prev_img.shape[0] // 4)
            template = prev_img[prev_img.shape[0]//2 - template_height//2:
                                prev_img.shape[0]//2 + template_height//2, :]
            
            result = cv2.matchTemplate(curr_img, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)
            
            if max_val > 0.7:
                mid_y_prev = prev_img.shape[0] // 2
                mid_y_curr = max_loc[1] + template_height // 2
                scroll_amount = abs(mid_y_prev - mid_y_curr)
        
        return scroll_amount
    
    def process_scroll_buffer(self):
        if len(self.scroll_buffer) < 2:
            return
        
        if self.final_image is None:
            self.final_image = self.scroll_buffer[0].copy()
            self.total_height = self.final_image.height
            self.current_y_offset = self.final_image.height
        
        for i in range(1, len(self.scroll_buffer)):
            current_img = self.scroll_buffer[i]
            overlap_amount = self.find_overlap(self.final_image, current_img)
            
            if overlap_amount > 0:
                new_content = current_img.crop((0, overlap_amount, current_img.width, current_img.height))
                new_height = self.total_height + new_content.height
                new_image = Image.new('RGB', (self.region[2], new_height))
                new_image.paste(self.final_image, (0, 0))
                new_image.paste(new_content, (0, self.total_height))
                self.final_image = new_image
                self.total_height = new_height
                print(f"Added {new_content.height} pixels of new content (overlap: {overlap_amount}px)")
        
        last_screenshot = self.scroll_buffer[-1]
        self.scroll_buffer = [last_screenshot]
    
    def find_overlap(self, final_img, next_img):
        final_array = np.array(final_img)
        next_array = np.array(next_img)
        
        bottom_height = min(150, final_array.shape[0] // 2)
        bottom_section = final_array[-bottom_height:, :]
        
        result = cv2.matchTemplate(next_array, bottom_section, cv2.TM_CCOEFF_NORMED)
        _, max_val, _, max_loc = cv2.minMaxLoc(result)
        
        if max_val > 0.7:
            overlap = max_loc[1] + bottom_section.shape[0]
            return min(overlap, next_array.shape[0] - 1)
        
        check_rows = min(100, final_array.shape[0] // 4, next_array.shape[0] // 4)
        for i in range(1, check_rows):
            bottom_row = final_array[-i, :]
            matches = []
            for j in range(min(100, next_array.shape[0])):
                top_row = next_array[j, :]
                similarity = np.mean(bottom_row == top_row)
                matches.append((similarity, j))
            best_match = max(matches, key=lambda x: x[0])
            if best_match[0] > 0.9:
                return best_match[1]
        
        return int(next_array.shape[0] * 0.2)
    
    def create_layer_mask(self):
        root = tk.Tk()
        root.attributes('-fullscreen', True)
        root.attributes('-alpha', 0.3)  
        root.configure(bg='black')
        
        canvas = tk.Canvas(root, bg='black', highlightthickness=0)
        canvas.pack(fill=tk.BOTH, expand=True)
        
        selection_rect = None
        start_x, start_y = 0, 0
        
        def on_press(event):
            nonlocal start_x, start_y, selection_rect
            if selection_rect:
                canvas.delete(selection_rect)
            start_x, start_y = event.x, event.y
            selection_rect = canvas.create_rectangle(
                start_x, start_y, start_x, start_y,
                outline='red', width=2
            )
        
        def on_drag(event):
            nonlocal selection_rect
            if selection_rect:
                canvas.coords(selection_rect, start_x, start_y, event.x, event.y)
        
        def on_release(event):
            nonlocal start_x, start_y, selection_rect
            x1, y1 = min(start_x, event.x), min(start_y, event.y)
            x2, y2 = max(start_x, event.x), max(start_y, event.y)
            self.region = (x1, y1, x2 - x1, y2 - y1)
            root.destroy()
        
        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        
        def on_escape(event):
            root.destroy()
        root.bind("<Escape>", on_escape)
        
        screen_size = pyautogui.size()
        instructions = "Click and drag to select the region for scrolling screenshot. Press ESC to cancel."
        canvas.create_text(
            screen_size[0]//2, 30,
            text=instructions,
            fill="white",
            font=("Arial", 16)
        )
        
        root.mainloop()
        
        return self.region is not None
    
    def save_final_image(self):
        if self.final_image is None:
            print("No content was captured.")
            return None
        
        final_image_with_drawings = self.apply_drawings()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        default_filename = f"scrolling_screenshot_{timestamp}.png"
        
        root = tk.Tk()
        root.withdraw()  
        file_path = filedialog.asksaveasfilename(
            initialdir=self.output_dir,
            initialfile=default_filename,
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")]
        )
        
        if not file_path:
            file_path = os.path.join(self.output_dir, default_filename)
        
        final_image_with_drawings.save(file_path)
        print(f"\nScrolling screenshot saved to: {file_path}")
        
        return file_path
    
    def apply_drawings(self):
        if not self.draw_lines or self.final_image is None:
            return self.final_image.copy()
        
        result = self.final_image.copy()
        draw = ImageDraw.Draw(result)
        
        for line, color in self.draw_lines:
            if len(line) > 1:
                draw.line(line, fill=color, width=2)
        
        return result
    
    def start_capture(self):
        print("\nScrolling Screenshot Capture Started")
        print("-----------------------------------")
        print(f"Selected region: {self.region}")
        print("Use your mouse wheel to scroll through content")
        print("Press 'ESC' to stop capturing and save")
        
        self.last_screenshot = None
        self.final_image = None
        self.scroll_buffer = []
        self.total_height = 0
        
        initial_screenshot = pyautogui.screenshot(region=self.region)
        self.scroll_buffer.append(initial_screenshot)
        self.last_screenshot = initial_screenshot
        
        self.is_capturing = True
        self.listener.start()
        
        while self.is_capturing:
            if keyboard.is_pressed('esc'):
                self.is_capturing = False
                print("\nCapture stopped. Processing screenshot...")
                break
            time.sleep(0.1)
        
        self.listener.stop()
        self.process_scroll_buffer()
        self.ask_to_draw()
    
    def ask_to_draw(self):
        if self.final_image is None:
            print("No image captured to draw on.")
            self.save_final_image()
            return
        
        print("\nDo you want to draw on the screenshot?")
        print("Press 'D' to draw, or 'S' to save without drawing")
        
        waiting_for_input = True
        while waiting_for_input:
            if keyboard.is_pressed('d'):
                waiting_for_input = False
                self.start_drawing_mode()
            elif keyboard.is_pressed('s'):
                waiting_for_input = False
                self.save_final_image()
            time.sleep(0.1)
    
    def select_draw_color(self):
        root = tk.Tk()
        root.withdraw() 
        color = colorchooser.askcolor(title="Choose drawing color", initialcolor=self.draw_color)
        if color[0]:
            self.draw_color = tuple(map(int, color[0]))
        return self.draw_color
    
    def start_drawing_mode(self):
        root = tk.Tk()
        root.title("Draw on Screenshot - Press 'ESC' to finish, 'C' to change color")
        
        screen_width, screen_height = root.winfo_screenwidth(), root.winfo_screenheight()
        img_width, img_height = self.final_image.width, self.final_image.height
        
        self.scale = 1.0
        self.display_img = ImageTk.PhotoImage(self.final_image)
        canvas = tk.Canvas(
            root,
            width=min(img_width, screen_width * 0.8),
            height=min(img_height, screen_height * 0.8),
            xscrollcommand=lambda *args: h_scrollbar.set(*args),
            yscrollcommand=lambda *args: v_scrollbar.set(*args)
        )
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        h_scrollbar = tk.Scrollbar(root, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        h_scrollbar.config(command=lambda *args: canvas.xview(*args))
        
        v_scrollbar = tk.Scrollbar(root)
        v_scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        v_scrollbar.config(command=lambda *args: canvas.yview(*args))
        
        canvas.create_image(0, 0, image=self.display_img, anchor=tk.NW)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        
        self.drawing_overlay = Image.new('RGBA', (img_width, img_height), (0, 0, 0, 0))
        self.draw_lines = []
        self.current_line = []
        self.is_drawing = False
        
        canvas_lines = []
        
        color_button = tk.Button(
            root, 
            text="Change Color", 
            command=lambda: self.update_draw_color(canvas),
            bg=self.rgb_to_hex(self.draw_color)
        )
        color_button.pack(pady=5)
        
        def start_line(event):
            nonlocal canvas_lines
            if self.is_drawing:
                return
            
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            img_x = int(canvas_x / self.scale)
            img_y = int(canvas_y / self.scale)
            
            self.current_line = [(img_x, img_y)]
            self.is_drawing = True
            
            canvas_lines.append([])
            canvas_lines[-1].append((canvas_x, canvas_y))
        
        def draw_line(event):
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            if not self.is_drawing or not canvas_lines:
                return
            
            last_point = canvas_lines[-1][-1]
            line_id = canvas.create_line(
                last_point[0], last_point[1],
                canvas_x, canvas_y,
                fill=self.rgb_to_hex(self.draw_color),
                width=2
            )
            canvas_lines[-1].append((canvas_x, canvas_y))
            
            img_x = int(canvas_x / self.scale)
            img_y = int(canvas_y / self.scale)
            self.current_line.append((img_x, img_y))
        
        def end_line(event):
            if not self.is_drawing or not self.current_line:
                return
            
            canvas_x = canvas.canvasx(event.x)
            canvas_y = canvas.canvasy(event.y)
            img_x = int(canvas_x / self.scale)
            img_y = int(canvas_y / self.scale)
            
            self.current_line.append((img_x, img_y))
            self.draw_lines.append((self.current_line.copy(), self.draw_color))
            self.current_line = []
            self.is_drawing = False
        
        canvas.bind("<ButtonPress-1>", start_line)
        canvas.bind("<B1-Motion>", draw_line)
        canvas.bind("<ButtonRelease-1>", end_line)
        
        def on_key(event):
            if event.keysym == 'Escape':
                self.quit_drawing_mode(root)
            elif event.keysym == 'c':
                self.update_draw_color(canvas, color_button)
        
        root.bind("<Key>", on_key)
        
        def zoom(event):
            factor = 1.1 if event.delta > 0 else 0.9
            self.scale *= factor
            self.redraw_canvas(canvas, factor)
        
        canvas.bind("<MouseWheel>", zoom)
        
        quit_button = tk.Button(root, text="Quit", command=lambda: self.quit_drawing_mode(root))
        quit_button.pack(pady=10)
        
        instructions = tk.Label(
            root, 
            text="Click and drag to draw. Press 'C' to change color. Use mouse wheel to zoom. Press 'ESC' or click Quit to finish."
        )
        instructions.pack(pady=5)
        
        root.mainloop()
    
    def update_draw_color(self, canvas, button=None):
        self.select_draw_color()
        if button:
            button.config(bg=self.rgb_to_hex(self.draw_color))
    
    def rgb_to_hex(self, rgb):
        return f'#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}'
    
    def redraw_canvas(self, canvas, factor):
        new_width = int(self.final_image.width * self.scale)
        new_height = int(self.final_image.height * self.scale)
        scaled_image = self.final_image.resize((new_width, new_height), Image.LANCZOS)
        self.display_img = ImageTk.PhotoImage(scaled_image)
        
        canvas.delete("all")
        canvas.create_image(0, 0, image=self.display_img, anchor=tk.NW)
        canvas.config(scrollregion=canvas.bbox(tk.ALL))
        self.update_canvas_lines(canvas)
    
    def update_canvas_lines(self, canvas):
        pass
    
    def quit_drawing_mode(self, root):
        root.destroy()
        self.save_final_image()
        exit()

def main():
    parser = argparse.ArgumentParser(description='Adaptive Scrolling Screenshot Tool')
    parser.add_argument('--output-dir', default='screenshots', help='Directory to save screenshots')
    args = parser.parse_args()
    
    tool = ScrollingScreenshotTool(output_dir=args.output_dir)
    
    print("Adaptive Scrolling Screenshot Tool")
    print("Press 'R' to start region selection")
    print("Press 'Q' to quit")
    
    while True:
        if keyboard.is_pressed('q'):
            print("Quitting...")
            break
        
        if keyboard.is_pressed('r'):
            time.sleep(0.2)
            print("Creating selection overlay...")
            if tool.create_layer_mask():
                tool.start_capture()
            else:
                print("Region selection was cancelled.")
        
        time.sleep(0.1)

if __name__ == "__main__":
    main()
