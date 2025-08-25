import cv2
import sys
import os
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def test_helmet_detection():
    """Test script to debug helmet detection on specific video frames"""
    
    try:
        from detectors.helmet_detector import create_helmet_detector
        print("✅ Helmet detector imported successfully")
    except ImportError as e:
        print(f"❌ Failed to import helmet detector: {e}")
        return
    
    # Initialize detector
    detector = create_helmet_detector()
    
    # Test with sample video
    video_path = "no_helmet_biker.webm"
    if not os.path.exists(video_path):
        print(f"❌ Video file not found: {video_path}")
        print("📝 Please make sure you have a video file named 'video_testing.mp4' in your project folder")
        return
    
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return
    
    print("🎥 Video opened successfully")
    print("🎮 Controls:")
    print("   - Press 'n' for next frame")
    print("   - Press 's' to select a motorcycle ROI")
    print("   - Press 'q' to quit")
    print("   - Press 'd' to toggle debug mode")
    
    frame_count = 0
    debug_mode = True
    selected_roi = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("❌ End of video or cannot read frame")
            break
        
        frame_count += 1
        
        # Display the frame
        display_frame = frame.copy()
        
        # Draw instructions on frame
        cv2.putText(display_frame, "Press 'n': Next frame, 's': Select ROI, 'd': Toggle debug, 'q': Quit", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(display_frame, f"Frame: {frame_count}, Debug: {'ON' if debug_mode else 'OFF'}", 
                   (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # If we have a selected ROI, test helmet detection on it
        if selected_roi is not None:
            x1, y1, x2, y2 = selected_roi
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 255), 3)
            cv2.putText(display_frame, "Selected ROI", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            
            # Test helmet detection on the selected ROI
            is_violation, confidence, violation_data, evidence_img = detector.detect_helmet_violation(
                frame, selected_roi, 999
            )
            
            # Display results
            status = "VIOLATION" if is_violation else "NO VIOLATION"
            color = (0, 0, 255) if is_violation else (0, 255, 0)
            cv2.putText(display_frame, f"{status} (Confidence: {confidence:.2f})", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
            
            print(f"Frame {frame_count}: {status}, Confidence: {confidence:.2f}")
            
            # If we have evidence image, show it
            if evidence_img is not None:
                cv2.imshow("Evidence", evidence_img)
        
        # Show the frame
        cv2.imshow("Helmet Detection Test", display_frame)
        
        # Handle keyboard input
        key = cv2.waitKey(0) & 0xFF
        
        if key == ord('q'):
            break
        elif key == ord('n'):
            continue  # Next frame
        elif key == ord('s'):
            # Let user select a region of interest
            print("🖱️ Select a motorcycle region and press ENTER or SPACE")
            roi = cv2.selectROI("Helmet Detection Test", display_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("ROI selector")
            
            if roi[2] > 0 and roi[3] > 0:  # If a valid ROI was selected
                x, y, w, h = roi
                selected_roi = [x, y, x + w, y + h]
                print(f"📐 Selected ROI: {selected_roi}")
            else:
                selected_roi = None
                print("❌ No ROI selected")
        elif key == ord('d'):
            debug_mode = not debug_mode
            # Toggle debug mode in the detector if possible
            if hasattr(detector, 'debug_mode'):
                detector.debug_mode = debug_mode
                print(f"🔧 Debug mode: {'ON' if debug_mode else 'OFF'}")
    
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Test completed")

def analyze_video_frames():
    """Analyze specific frames from the video to understand what's happening"""
    
    cap = cv2.VideoCapture("video_testing.mp4")
    
    if not cap.isOpened():
        print("❌ Cannot open video file")
        return
    
    # Get total frames and FPS
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"📊 Video analysis:")
    print(f"   - Total frames: {total_frames}")
    print(f"   - FPS: {fps:.1f}")
    print(f"   - Duration: {total_frames/fps:.1f} seconds")
    
    # Analyze every 100th frame
    for frame_num in range(0, total_frames, 100):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_num)
        ret, frame = cap.read()
        
        if ret:
            # Save the frame for analysis
            frame_filename = f"debug/frame_{frame_num:05d}.jpg"
            os.makedirs("debug", exist_ok=True)
            cv2.imwrite(frame_filename, frame)
            print(f"💾 Saved frame {frame_num} to {frame_filename}")
    
    cap.release()
    print("✅ Frame analysis completed. Check the 'debug' folder for saved frames.")

if __name__ == "__main__":
    print("🔧 Helmet Detection Test Script")
    print("1. Test helmet detection on video")
    print("2. Analyze video frames")
    
    choice = input("Select option (1 or 2): ").strip()
    
    if choice == "1":
        test_helmet_detection()
    elif choice == "2":
        analyze_video_frames()
    else:
        print("❌ Invalid choice")