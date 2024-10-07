# Document for tracking DMD commands progress and missing features

DMD is controlled by compiled C scripts. They work relying on a binarised (BIN) set of images which all have an unique ID.

By providing the C script with a VEC file [ a .txt file containing the order of image-IDs to be presented and their duration ], a command is sent to the DMD which shows the image and sends a trigger signal to a dedicated channel of the MEA.

For a detailed list of tasks and progress, see the [TODO.md](../TODO.md) file.