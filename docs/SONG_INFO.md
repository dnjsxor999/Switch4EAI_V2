## Dance Timing Information

| Song | Padding (Time-to-Dance in seconds) | Dance Duration (seconds) |
| --- | --- | --- |
| Old_Town_Road | 4.3 (11.0 *for online*) | 161 (155 *for online*) |
| Heart_Of_Glass | 8.0 | 216 |
| Unstoppable | 12.2 | 204 |
| Padam_Padam | 24.5 | 149 |
| Pink_Venom | 15.0 | 178 |


### Note
The padding (wait time) is not a fixed value; it may vary depending on your network conditions and the current status of Nintendo’s servers. Please adjust the padding time as needed for your setup, usually within a range of ±1 second.

You can use `scripts/preprocess/pad_gmr_pickles.py` to apply custom padding to the songs. Note that for `Old_Town_Road`, offline motion is padded with 4.3 seconds, but should wait 11 seconds when running online experiment.