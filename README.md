# Image Restoration using Partial Differential Equations

Equations originating from physics have recently found their way to other areas. One, possibly surprising, application is that of restoring art or images (image restoration, or, image inpainting).

To understand how Partial Differential Equations (PDEs) from physics can help in image restoration, consider some grayscale image of your own choice. Due to graffiti painters, let us assume that a piece is missing (black region). Can we fill in the missing region without any information of what is missing? This may seem like a hopeless task, but PDEs are here to help!

PDE-based methods for image restoration are based on propagating the information (typically, intensity values and gradients) at the boundaries of the missing region inwards. The propagation is performed by solving a partial differential equations with specified boundary conditions.

Check out the results on [YouTube](https://youtu.be/NSAQF8PqJYM)!
[![Image Restoration using Partial Differential Equations](https://joebinns.com/documents/fake_thumbnails/image_restoration_thumbnail_time.png)](https://youtu.be/NSAQF8PqJYM "Image Restoration using Partial Differential Equations. Click to watch.")

Collaborators: [Joshua Greaves](https://github.com/jo6202gr-s), [Daniel Larsson Persson](https://github.com/Dhanari), [Joe Binns](https://joebinns.com/).
