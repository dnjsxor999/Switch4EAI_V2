"""User input handling utilities."""


def get_user_timing_inputs() -> tuple[float, float]:
    """
    Prompt user for wait time and song duration.
    
    Returns:
        (wait_time, song_duration) in seconds
    """
    print("\nYou should Enter wait time(s) and song duration(s) for the current song:")
    
    while True:
        try:
            wait_time = float(input("Enter Wait Time: "))
            if wait_time < 0:
                print("Wait time must be non-negative. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    while True:
        try:
            song_duration = float(input("Enter Song Duration: "))
            if song_duration <= 0:
                print("Song duration must be positive. Please try again.")
                continue
            break
        except ValueError:
            print("Invalid input. Please enter a number.")
    
    print('Press enter again, just as you press the "a" button on the Switch Controller...')
    input()
    
    return wait_time, song_duration

