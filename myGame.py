# import cv2
# import pyautogui
# from time import time
# from myPose import myPose


# class myGame():
#     def __init__(self):
#         self.pose = myPose()

#         # Initialize a variable to store the state of the game (started or not).
#         self.game_started = False
#         # Initialize a variable to store the index of the current horizontal position of the person.
#         # At Start the character is at center so the index is 1 and it can move left (value 0) and right (value 2).
#         self.x_pos_index = 1

#         # Initialize a variable to store the index of the current vertical posture of the person.
#         # At Start the person is standing so the index is 1 and he can crouch (value 0) and jump (value 2).
#         self.y_pos_index = 1

#         # Initialize a counter to store count of the number of consecutive frames with person's hands joined.
#         self.counter = 0

#         # Initialize a variable to store the time of the previous frame.
#         self.time1 = 0

#         # Declate a variable to store the intial y-coordinate of the mid-point of both shoulders of the person.
#         self.MID_Y = None

#         # Initialize the number of consecutive frames on which we want to check if person hands joined before starting the game.
#         self.num_of_frames = 10

#     def move_LRC(self, LRC):
#         # Check if the person has moved to left from center or to center from right.
#         if (LRC == 'Left' and self.x_pos_index != 0) or (
#                 LRC == 'Center' and self.x_pos_index == 2):

#             # Press the left arrow key.
#             pyautogui.press('left')

#             # Update the horizontal position index of the character.
#             self.x_pos_index -= 1

#             # Check if the person has moved to Right from center or to center from left.
#         elif (LRC == 'Right' and self.x_pos_index != 2) or (
#                 LRC == 'Center' and self.x_pos_index == 0):

#             # Press the right arrow key.
#             pyautogui.press('right')

#             # Update the horizontal position index of the character.
#             self.x_pos_index += 1
#         return

#     def move_JSD(self, JSD):
#         # Check if the person has jumped.
#         if JSD == 'Jumping' and self.y_pos_index == 1:

#             # Press the up arrow key
#             pyautogui.press('up')

#             # Update the veritcal position index of  the character.
#             self.y_pos_index += 1

#             # Check if the person has crouched.
#         elif JSD == 'Crouching' and self.y_pos_index == 1:

#             # Press the down arrow key
#             pyautogui.press('down')

#             # Update the veritcal position index of the character.
#             self.y_pos_index -= 1

#         # Check if the person has stood.
#         elif JSD == 'Standing' and self.y_pos_index != 1:

#             # Update the veritcal position index of the character.
#             self.y_pos_index = 1
#         return

#     def play(self):

#         # Initialize the VideoCapture object to read from the webcam.
#         cap = cv2.VideoCapture(0)
#         cap.set(3, 1280)
#         cap.set(4, 960)

#         # Create named window for resizing purposes.
#         cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)

#         while True:
#             # Read a frame
#             ret, image = cap.read()

#             # Check if frame is not read properly then continue to the next iteration to read the next frame.
#             if not ret:
#                 continue
#             else:
#                 # Flip the frame horizontally for natural (selfie-view) visualization.
#                 image = cv2.flip(image, 1)

#                 # Get the height and width of the frame of the webcam video.
#                 image_height, image_width, _ = image.shape

#                 # Perform the pose detection on the frame.
#                 image, results = self.pose.detectPose(image, self.pose.pose_video, draw=self.game_started)

#                 # Check if the pose landmarks in the frame are detected.
#                 if results.pose_landmarks:

#                     # Check if the game has started
#                     if self.game_started:
#                         # Commands to control the horizontal movements of the character.
#                         # --------------------------------------------------------------------------------------------------------------

#                         # Get horizontal position of the person in the frame.
#                         image, LRC = self.pose.checkPose_LRC(image, results, draw=True)
#                         self.move_LRC(LRC)

#                         # ------------------------------------------------------------------------------------------------------------------

#                         # Commands to control the vertical movements of the character.
#                         # ------------------------------------------------------------------------------------------------------------------

#                         # Check if the intial y-coordinate of the mid-point of both shoulders of the person has a value.
#                         if self.MID_Y:
#                             image, JSD = self.pose.checkPose_JSD(image, results, self.MID_Y, draw=True)
#                             self.move_JSD(JSD)

#                     # Otherwise if the game has not started
#                     else:

#                         # Write the text representing the way to start the game on the frame.
#                         cv2.putText(image, 'JOIN BOTH HANDS TO START THE GAME.', (5, image_height - 10),
#                                     cv2.FONT_HERSHEY_PLAIN,
#                                     2, (0, 255, 0), 3)

#                     # Command to Start or resume the game.
#                     # ------------------------------------------------------------------------------------------------------------------

#                     # Check if the left and right hands are joined.
#                     if self.pose.checkHandsJoined(image, results)[1] == 'Hands Joined':

#                         # Increment the count of consecutive frames with +ve condition.
#                         self.counter += 1

#                         # Check if the counter is equal to the required number of consecutive frames.
#                         if self.counter == self.num_of_frames:

#                             # Command to Start the game first time.
#                             # ----------------------------------------------------------------------------------------------------------

#                             # Check if the game has not started yet.
#                             if not (self.game_started):

#                                 # Update the value of the variable that stores the game state.
#                                 self.game_started = True

#                                 # Retreive the y-coordinate of the left shoulder landmark.
#                                 left_y = int(results.pose_landmarks.landmark[
#                                                  self.pose.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)

#                                 # Retreive the y-coordinate of the right shoulder landmark.
#                                 right_y = int(results.pose_landmarks.landmark[
#                                                   self.pose.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)

#                                 # Calculate the intial y-coordinate of the mid-point of both shoulders of the person.
#                                 self.MID_Y = abs(right_y + left_y) // 2

#                                 # Move to 1300, 800, then click the left mouse button to start the game.
#                                 pyautogui.click(x=1300, y=800, button='left')

#                             # ----------------------------------------------------------------------------------------------------------

#                             # Command to resume the game after death of the character.
#                             # ----------------------------------------------------------------------------------------------------------

#                             # Otherwise if the game has started.
#                             else:

#                                 # Press the space key.
#                                 pyautogui.press('space')

#                             # ----------------------------------------------------------------------------------------------------------
#                             # Update the counter value to zero.
#                             self.counter = 0

#                     # Otherwise if the left and right hands are not joined.
#                     else:

#                         # Update the counter value to zero.
#                         self.counter = 0

#                 # Otherwise if the pose landmarks in the frame are not detected.
#                 else:
#                     self.counter = 0

#                 # Calculate the frames updates in one second
#                 # ----------------------------------------------------------------------------------------------------------------------

#                 # Set the time for this frame to the current time.
#                 time2 = time()

#                 # Check if the difference between the previous and this frame time > 0 to avoid division by zero.
#                 if (time2 - self.time1) > 0:

#                     # Calculate the number of frames per second.
#                     frames_per_second = 1.0 / (time2 - self.time1)

#                     # Write the calculated number of frames per second on the frame.
#                     cv2.putText(image, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN,
#                                 2, (0, 255, 0), 3)

#                 # Update the previous frame time to this frame time.
#                 # As this frame will become previous frame in next iteration.
#                 self.time1 = time2

#                 cv2.imshow("Subway Surfers with Pose Detection", image)

#                 # Wait for 1ms. If a a key is pressed, check
#                 if cv2.waitKey(1) == ord('q'):
#                     break

#         cap.release()
#         cv2.destroyAllWindows()


# myGame = myGame()
# myGame.play()
import cv2  # Thư viện OpenCV để xử lý hình ảnh và video.
import pyautogui  # Thư viện để tự động hóa các hành động chuột và bàn phím.
from time import time  # Hàm để tính thời gian.
from myPose import myPose  # Lớp myPose để phát hiện các điểm mốc tư thế.

# Lớp myGame để thực hiện trò chơi với điều khiển bằng tư thế.
class myGame():
    def __init__(self):
        """
        Khởi tạo trò chơi, bao gồm các thuộc tính để theo dõi tư thế và trạng thái của trò chơi.
        """
        self.pose = myPose()  # Khởi tạo đối tượng myPose để phát hiện tư thế.
        self.game_started = False  # Trạng thái trò chơi, mặc định chưa bắt đầu.
        self.x_pos_index = 1  # Vị trí ngang của nhân vật (0: trái, 1: giữa, 2: phải).
        self.y_pos_index = 1  # Vị trí dọc của nhân vật (0: ngồi xổm, 1: đứng, 2: nhảy).
        self.counter = 0  # Đếm số khung hình liên tiếp tay người chơi đang chạm nhau.
        self.time1 = 0  # Thời gian của khung hình trước.
        self.MID_Y = None  # Tọa độ y ban đầu của điểm giữa hai vai.
        self.num_of_frames = 10  # Số khung hình liên tiếp tay người chơi phải chạm nhau để bắt đầu trò chơi.

    def move_LRC(self, LRC):
        """
        Di chuyển nhân vật theo hướng ngang (trái, phải, giữa) dựa trên tư thế của người chơi.
        """
        if (LRC == 'Left' and self.x_pos_index != 0) or (LRC == 'Center' and self.x_pos_index == 2):
            # Di chuyển sang trái.
            pyautogui.press('left')
            self.x_pos_index -= 1  # Cập nhật vị trí ngang của nhân vật.
            print(f"Đang di chuyển: {LRC} - Phím 'left' đã được nhấn.")
        elif (LRC == 'Right' and self.x_pos_index != 2) or (LRC == 'Center' and self.x_pos_index == 0):
            # Di chuyển sang phải.
            pyautogui.press('right')
            self.x_pos_index += 1  # Cập nhật vị trí ngang của nhân vật.
            print(f"Đang di chuyển: {LRC} - Phím 'right' đã được nhấn.")
        return

    def move_JSD(self, JSD):
        """
        Di chuyển nhân vật theo hướng dọc (nhảy, ngồi xổm, đứng thẳng) dựa trên tư thế của người chơi.
        """
        if JSD == 'Jumping' and self.y_pos_index == 1:
            # Di chuyển nhân vật lên (nhảy).
            pyautogui.press('up')
            self.y_pos_index += 1  # Cập nhật vị trí dọc của nhân vật.
            print(f"Đang di chuyển: {JSD} - Phím 'up' đã được nhấn.")
        elif JSD == 'Crouching' and self.y_pos_index == 1:
            # Di chuyển nhân vật xuống (ngồi xổm).
            pyautogui.press('down')
            self.y_pos_index -= 1  # Cập nhật vị trí dọc của nhân vật.
            print(f"Đang di chuyển: {JSD} - Phím 'down' đã được nhấn.")
        elif JSD == 'Standing' and self.y_pos_index != 1:
            # Nếu người chơi đứng thẳng, đặt lại vị trí dọc của nhân vật.
            self.y_pos_index = 1
            print(f"Đang di chuyển: {JSD} - Phím 'Standing' đã được nhấn.")
        return

    def play(self):
        """
        Hàm chính để chơi trò chơi. Liên tục nhận khung hình từ webcam và phát hiện tư thế để điều khiển nhân vật.
        """
        cap = cv2.VideoCapture(0)  # Mở webcam.
        cap.set(3, 1280)  # Cài đặt độ phân giải ngang.
        cap.set(4, 960)  # Cài đặt độ phân giải dọc.

        cv2.namedWindow('Subway Surfers with Pose Detection', cv2.WINDOW_NORMAL)  # Tạo cửa sổ hiển thị.

        while True:
            ret, image = cap.read()  # Đọc một khung hình từ webcam.
            if not ret:
                continue  # Nếu không đọc được khung hình, tiếp tục vòng lặp.
            else:
                image = cv2.flip(image, 1)  # Lật hình ảnh để hiển thị giống góc nhìn selfie.

                image_height, image_width, _ = image.shape  # Lấy kích thước của hình ảnh.

                # Phát hiện tư thế trong khung hình.
                image, results = self.pose.detectPose(image, self.pose.pose_video, draw=self.game_started)

                if results.pose_landmarks:  # Kiểm tra nếu các điểm mốc tư thế được phát hiện.
                    if self.game_started:
                        # Điều khiển di chuyển ngang.
                        image, LRC = self.pose.checkPose_LRC(image, results, draw=True)
                        self.move_LRC(LRC)

                        # Điều khiển di chuyển dọc (nhảy, ngồi xổm, đứng thẳng).
                        if self.MID_Y:
                            image, JSD = self.pose.checkPose_JSD(image, results, self.MID_Y, draw=True)
                            self.move_JSD(JSD)

                    else:
                        # Nếu trò chơi chưa bắt đầu, hiển thị yêu cầu bắt đầu trò chơi.
                        cv2.putText(image, 'JOIN BOTH HANDS TO START THE GAME.', (5, image_height - 10),
                                    cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

                    # Kiểm tra xem người chơi đã chạm tay để bắt đầu trò chơi chưa.
                    if self.pose.checkHandsJoined(image, results)[1] == 'Hands Joined':
                        self.counter += 1  # Tăng bộ đếm khung hình liên tiếp tay người chơi đang chạm nhau.
                        if self.counter == self.num_of_frames:  # Nếu đủ số khung hình liên tiếp, bắt đầu trò chơi.
                            if not self.game_started:
                                self.game_started = True  # Bắt đầu trò chơi.

                                # Tính tọa độ y của điểm giữa hai vai để phục vụ điều khiển nhảy/ngồi xổm.
                                left_y = int(results.pose_landmarks.landmark[self.pose.mp_pose.PoseLandmark.RIGHT_SHOULDER].y * image_height)
                                right_y = int(results.pose_landmarks.landmark[self.pose.mp_pose.PoseLandmark.LEFT_SHOULDER].y * image_height)
                                self.MID_Y = abs(right_y + left_y) // 2  # Tính tọa độ y của điểm giữa hai vai.

                                # Nhấn chuột để bắt đầu trò chơi.
                                pyautogui.click(x=1300, y=800, button='left')
                            else:
                                # Nếu trò chơi đã bắt đầu, nhấn phím Space để tiếp tục trò chơi.
                                pyautogui.press('space')

                            self.counter = 0  # Đặt lại bộ đếm.

                    else:
                        self.counter = 0  # Nếu tay không chạm, đặt lại bộ đếm.

                else:
                    self.counter = 0  # Nếu không phát hiện được các điểm mốc tư thế, đặt lại bộ đếm.

                # Tính toán số khung hình mỗi giây (FPS).
                time2 = time()
                if (time2 - self.time1) > 0:  # Đảm bảo không chia cho 0.
                    frames_per_second = 1.0 / (time2 - self.time1)  # Tính số khung hình mỗi giây.
                    cv2.putText(image, 'FPS: {}'.format(int(frames_per_second)), (10, 30), cv2.FONT_HERSHEY_PLAIN, 2, (0, 255, 0), 3)

                self.time1 = time2  # Cập nhật thời gian của khung hình này để tính cho khung hình tiếp theo.

                cv2.imshow("Subway Surfers with Pose Detection", image)  # Hiển thị khung hình.

                # Chờ 1ms, nếu nhấn phím 'q' thì thoát.
                if cv2.waitKey(1) == ord('q'):
                    break

        cap.release()  # Giải phóng tài nguyên webcam.
        cv2.destroyAllWindows()  # Đóng tất cả các cửa sổ OpenCV.

# Tạo đối tượng trò chơi và bắt đầu chơi.
myGame = myGame()
myGame.play()
