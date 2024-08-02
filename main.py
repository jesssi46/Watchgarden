#importing the required libraries
import sensor, image, time, tf
from machine import Pin
 
# Define the pins using the correct port names
# If you used other pins for your components
# you have to use their corresponding port name as defined in the official Arduino Portenta H7 pinout layout.
SPEAKER_PIN = 'PC7'  # Corresponds to digital pin 4
MOTION_PIN = 'PG7'   # Corresponds to digital pin 3
 
# Initialize the motion sensor and speaker
motion_sensor = Pin(MOTION_PIN, Pin.IN)
speaker = Pin(SPEAKER_PIN, Pin.OUT)
 
def beep(freq, duration, duty_cycle=0.5):
    period = 1.0 / freq
    cycles = int(freq * (duration / 1000.0))
    on_time = period * duty_cycle
    off_time = period * (1 - duty_cycle)
    for _ in range(cycles):
        speaker.on()
        time.sleep(on_time)
        speaker.off()
        time.sleep(off_time)
 
def short_beep():
    beep(1000, 400)  # Beep at 1000 Hz for 400 ms
 
# Initialize the camera sensor
sensor.reset()
sensor.set_pixformat(sensor.GRAYSCALE) # Set pixel format to GRAYSCALE
sensor.set_framesize(sensor.QVGA)      # Set frame size to QVGA (320x240)
sensor.set_windowing((96, 96))         # Set 96x96 window
sensor.skip_frames(time=2000)          # Let the camera adjust
 
print("Camera initialized")
net = "trained.tflite"
labels = [line.rstrip('\n') for line in open("labels.txt")]
print("Model loaded")
clock = time.clock()
print("Ready")
 
motion_detected = False
motion_debounce_time = 2000  # 2 seconds debounce time
last_motion_time = 0
cooldown_time = 5000  # 10 seconds cooldown after detection
 
while True:
    current_time = time.ticks_ms()
    if motion_sensor.value() and not motion_detected:
        print("Motion detected at", time.localtime())
        motion_detected = True
        last_motion_time = current_time
 
        clock.tick()
        img = sensor.snapshot()
        img.rotation_corr(z_rotation=90)  # Rotate the image 90 degrees to the right
        print("Image captured")
 
        # Run inference
        for obj in tf.classify(net, img, min_scale=1.0, scale_mul=0.8, x_overlap=0.5, y_overlap=0.5):
            print("****\nPredictions at [x=%d,y=%d,w=%d,h=%d]" % obj.rect())
            img.draw_rectangle(obj.rect())
            predictions_list = list(zip(labels, obj.output()))
 
 
            for label, confidence in predictions_list:
                print("%s = %f" % (label, confidence))
                if confidence > 0.85 and label == "Bird":  # Check for Bird with at least 85% confidence
                    print("It's a", label, "!")
                    short_beep()  # Play a short beep sound
                    time.sleep(cooldown_time / 1000)  # Wait for the cooldown period before checking for motion again
            print("Ready to run loop again")
 
    elif motion_detected and (time.ticks_diff(current_time, last_motion_time) > motion_debounce_time):
        motion_detected = False
 
    time.sleep(0.2)  # Small delay to avoid busy-waiting