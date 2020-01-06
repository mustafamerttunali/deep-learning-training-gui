import Augmentor

test = "True"

test2 = "True"

test3 = "True"

imgaug = "True"

if imgaug == "True":

    p = Augmentor.Pipeline("images/", output_directory="")

    if test == "True":
        print("Rotate kismindayim")
        p.rotate(probability=0.7, max_left_rotation=10, max_right_rotation=10)

    if test2 == "True":
        print("zoom kismindayim")
        p.zoom(probability=0.5, min_factor=1.1, max_factor=1.5)

    if test3 == "True":
        print("flip kismindayim")
        p.flip_left_right(probability = 1)
        
    p.sample(50) 
    p.process()