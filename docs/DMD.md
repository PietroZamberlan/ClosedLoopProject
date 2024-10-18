# Document for tracking DMD commands progress and missing features

DMD is controlled by compiled C scripts. They work relying on a binarised (BIN) set of images which all have an unique ID.

By providing the C script with a VEC file [ a .txt file containing the order of image-IDs to be presented and their duration ], a command is sent to the DMD which shows the image and sends a trigger signal to a dedicated channel of the MEA.

For a detailed list of tasks and progress, see the [TODO.md](../TODO.md) file.

## Issues
1. The DMD seems to not be sending the amount of triggers that i select as the number of files in the VEC file
    It sends a round number when sending less than ~25 triggers ( it sends about 30, but they are at the end )