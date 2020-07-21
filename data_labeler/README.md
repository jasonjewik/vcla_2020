# Example usage and output

## Set up:

```
$ data_path=/home/jason/Documents/drivedatanew
$ darknet_path=/home/jason/GitRepos/darknet-pjreddie
$ conda activate data_labeler
```

## Starting the program

```
(data_labeler) $ python label_data.py $data_path $darknet_path
Detected OS: linux
generating pickles
Progress: [ ########## ] 100.0%
successfully wrote pkl files to /home/jason/Documents/drivedatanew/pickles
reading pickle files
Progress: [ ########## ] 100.0%
read all pickle files
generating images
Progress: [ ########## ] 100.0%
successfully converted to images, stored in /home/jason/Documents/drivedatanew/imvid
Progress: [ ########## ] 100.0%
successfully cropped images, stored in /home/jason/Documents/drivedatanew/crop
```

## Darknet

I truncated the Darknet output because it is very long, but you should see something like this. The config file errors are OK and should not break anything if you use my Darknet fork.

```
Config file error line 2, could parse: train
Config file error line 3, could parse: valid
Config file error line 5, could parse: backup
.
.
.
Cannot load image "data/labels/124_7.png"
Cannot load image "data/labels/125_7.png"
Cannot load image "data/labels/126_7.png"
.
.
.
layer     filters    size              input                output
    0 conv     32  3 x 3 / 1   608 x 608 x   3   ->   608 x 608 x  32  0.639 BFLOPs
    1 conv     64  3 x 3 / 2   608 x 608 x  32   ->   304 x 304 x  64  3.407 BFL
.
.
.
Enter Image Path: /home/jason/Documents/drivedatanew/crop/000000928_img.jpg: Predicted in 0.040640 seconds.
Output fname: /home/jason/Documents/drivedatanew/crop/000000928_img.txt
motorcycle: 100%

```

## Getting final results

```
Progress: [ ########## ] 100.0%
more drive data than image data
created directory /home/jason/Documents/drivedatanew/results
generating pickled results
done writing results to /home/jason/Documents/drivedatanew/results
all data has been labeled
cleaning up intermediate files
done with cleanup
```
