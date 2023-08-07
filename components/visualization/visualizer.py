from os import mkdir, path, remove, rmdir
from audio2numpy import open_audio
from joblib import Parallel, cpu_count, delayed
from multiprocessing import Manager
import matplotlib.pyplot as plt
import numpy as np
import cv2
from numpy.typing import NDArray
import subprocess
from sys import exit, stdout, stderr

channel = "avarage"
framerate = 30
duration = framerate/1000
bins_amount = 64
xlog = 0
ylog = 0
DEFAULT_CHUNKSIZE = 128
VID_CODEC = 'mp4v'
VID_EXT = '.mp4'
end = -1

def load_audio(audio_file):
    if not path.exists(audio_file):
        raise FileNotFoundError("Audio file does not exist")
    else:
        waveform, sample_rate = open_audio(audio_file)
        print("waveform: " + str(waveform))
        print("sample_rate: " + str(sample_rate))
        return waveform, sample_rate

def calculate_frameData(fileData, samplerate):
    channel = "avarage"
    framerate = 30
    duration = 1000/framerate
    start = 0
    global end
    end = len(fileData)/samplerate

    channels = []

    if len(fileData.shape) > 1:
        if channel == "avarage":
            channels.append(np.mean(fileData, axis=1))
        elif channel == "left":
            channels.append(fileData[:,0])
        elif channel == "right":
            channels.append(fileData[:,1])
        else:
            for i in range(fileData.shape[1]):
                channels.append(fileData[:,i])
    else:
        channels.append(fileData)

    print("channels: " + str(channels))
    
    frameData = []
    a = 0
    for channel in channels:
        if a == 0:
            print("channel: " + str(channel))

        channelData = channel[int(start*samplerate):int(end*samplerate)]

        if a == 0:
            print("channelData: " + str(channelData))

        channelFrameData = []
        stepSize = samplerate/framerate
        print("samplerate: " + str(samplerate))
        print("framerate: " + str(framerate))
        print("stepSize: " + str(stepSize))
        b = 0
        for i in range(int(np.ceil(len(channelData)/stepSize))):
            frameDataMidpoint = stepSize*i + (stepSize/2)
            frameDataStart = int(frameDataMidpoint - (duration/1000/2)*samplerate)
            frameDataEnd = int(frameDataMidpoint + (duration/1000/2)*samplerate)

            if a == 0 and b == 0:
                print("frameDataMidpoint: " + str(frameDataMidpoint))
                print("duration: " + str(duration))
                print("frameDataStart: " + str(frameDataStart))
                print("frameDataEnd: " + str(frameDataEnd))

            if frameDataStart < 0:
                emptyFrame = np.zeros(int(duration/1000 * samplerate))
                currentFrameData = channelData[0:frameDataEnd]
                emptyFrame[0:len(currentFrameData)] = currentFrameData
                currentFrameData = emptyFrame
            elif frameDataEnd > len(channelData):
                emptyFrame = np.zeros(int(duration/1000 * samplerate))
                currentFrameData = channelData[frameDataStart:]
                emptyFrame[0:len(currentFrameData)] = currentFrameData
                currentFrameData = emptyFrame
            else:
                currentFrameData = channelData[int(frameDataStart):int(frameDataEnd)]
            
            if a == 0 and b == 0:
                print("currentFrameData: " + str(currentFrameData))

            frameDataAmplitudes = abs(np.fft.rfft(currentFrameData))

            frameDataAmplitudes = frameDataAmplitudes[int(0/(samplerate/2)*len(frameDataAmplitudes)):int((samplerate/2)/(samplerate/2)*len(frameDataAmplitudes))]

            channelFrameData.append(frameDataAmplitudes)
            #print("channelFrameData: " + str(channelFrameData[0]))
            b += 1

        frameData.append(channelFrameData)
        #print("frameData: " + str(frameData[0]))
        a += 1

    return frameData

def create_bins(frameData):
    #print("frameData: " + str(len(frameData)))
    bins = []
    a = 0
    for channel in frameData:
        if a == 0:
            print("channel: " + str(channel[0][0]))
        channelBins = []
        b = 0
        for data in channel:
            if a == 0 and b == 0:
                print("data: " + str(data[0]))
            frameBins = []
            c = 0
            for i in range(bins_amount):
                if xlog == 0:
                    dataStart = int(i * len(data) / bins_amount)
                    dataEnd = int((i + 1) * len(data) / bins_amount)
                else:
                    dataStart = int((i/bins_amount)**xlog * len(data))
                    dataEnd = int(((i+1)/bins_amount)**xlog * len(data))
                if dataStart == dataEnd:
                    dataEnd += 1
                frameBins.append(np.mean(data[dataStart:dataEnd]))
                #print("frameBins: " + str(frameBins[0]))
                if a == 0 and b == 0 and c == 0:
                    print("dataStart: " + str(dataStart))
                    print("dataEnd: " + str(dataEnd))
                    print("frameBins: " + str(frameBins))
                c += 1
            channelBins.append(frameBins)

            #print("channelBins: " + str(channelBins[0]))
            b += 1
        
        bins.append(channelBins)

        #print("bins: " + str(bins[0]))
        a += 1

    return bins

def render_save_frames(bins):
    ylog = 0
    processes = cpu_count()
    chunkSize = int(DEFAULT_CHUNKSIZE / processes)

    bins = bins/np.max(bins)

    if ylog != 0:
        div = np.log2(ylog + 1)
        bins = np.log2(ylog * np.array(bins) + 1) / div
    
    numChunks = int(np.ceil(bins.shape[1]/(processes * chunkSize))) * processes

    shMem = Manager().dict()
    shMem["framecount"] = 0
    Parallel(n_jobs=processes)(delayed(render_save_partial)(j, numChunks, bins, shMem) for j in range(processes))

    print_progress_bar(bins.shape[1], bins.shape[1])
    print()

def render_save_partial(partialCounter, numChunks, bins, shMem):
    imageSequence = False
    destination = "output"
    framerate = 30
    width = 1920
    height = 540
    processes = cpu_count()

    if imageSequence:
        vid = None
    else:
        fourcc = cv2.VideoWriter_fourcc(*VID_CODEC)
        dest = destination+"/part"+str(partialCounter)+VID_EXT
        vid = cv2.VideoWriter(dest, fourcc, framerate, (width, height))

    chunksPerProcess = int(numChunks/processes)
    for i in range(chunksPerProcess):
        chunkCounter = partialCounter * chunksPerProcess + i
        render_save_chunk(chunkCounter, numChunks, bins, vid, shMem)
    
    if not imageSequence:
        vid.release()

def render_save_chunk(chunkCounter, numChunks, bins, vid, shMem):
    processes = cpu_count()
    chunkSize = int(DEFAULT_CHUNKSIZE / processes)
    imageSequence = False
    destination = "output"

    chunksPerProcess = int(numChunks/processes)
    finishedChunkSets = int(chunkCounter/chunksPerProcess)
    framePerProcess = int(bins.shape[1]/processes)
    currentChunkNumInNewSet = chunkCounter - finishedChunkSets * chunksPerProcess
    start = finishedChunkSets * framePerProcess + currentChunkNumInNewSet * chunkSize
    end = start + chunkSize

    if chunkCounter % chunksPerProcess == chunksPerProcess - 1:
        completeChunkSets = int(numChunks/processes) - 1
        fullSetChunks = completeChunkSets * processes
        fullSetFrames = fullSetChunks * chunkSize
        remainingFrames = bins.shape[1] - fullSetFrames
        remainingChunkSize = int(remainingFrames / processes)
        end = start + remainingChunkSize
    
    frames = render_chunk_frames(bins, start, end)
    for i in range(len(frames)):
        if imageSequence:
            plt.imsave(str(destination) + "/" + str(start + i) + ".png", frames[i], vmin=0, vmax=255, cmap="gray")
        else:
            vid.write(frames[i])
        shMem["framecount"] += 1
        print_progress_bar(shMem["framecount"], bins.shape[1])

def render_chunk_frames(bins, start, end):
    frames = []
    for j in range(start, end):
        frame = render_frame(bins, j)
        frames.append(frame)
    return frames

def render_frame(bins, j):
    if len(bins) == 1:
        bins = bins[0]
        frame = render_mono_channel(bins, j)
    if len(bins) == 2:
        frame = render_stereo_channel(bins, j)
    
    return frame

def render_mono_channel(bins, j):
    mirror = 0
    height = 540
    width = 1920
    backgroundColor = hex2rgb("000000")
    radial = True
    style = "fill"
    barHeight = -1
    binWidth = width/bins * (5/6)
    binSpacing = width/bins * (1/6)
    color = hex2rgb("ffffff")
    lineThickness = 3
    radiusStart = height/6
    radiusEnd = height/2

    if mirror == 0:
        this_height = height
    else:
        fullFrame = np.full((height, width, 3), backgroundColor).astype(np.uint8)
        this_height = int(height/2)
    
    frame = np.full((this_height, width, 3), backgroundColor).astype(np.uint8)

    if not radial:
        if style == "bars" and barHeight == -1:
            for k in range(bins):
                frame[:int(np.ceil(bins[j,k]*this_height)),
                      int(k/bins*width + binSpacing/2):int((k+1)/bins*width - binSpacing/2)] = color

        if style == "bars" and barHeight != -1 or style == "circles" or style == "donuts":
            point = renderPoint(bins)
            binSpace = this_height - point.shape[0]

            for k in range(bins):
                binStart = int(k*binWidth + k*binSpacing)
                offset = int(binWidth - point.shape[1])

                frame[int(bins[j,k]*binSpace):int(bins[j,k]*binSpace + point.shape[0]),
                      int(k/bins*width + binSpacing/2):int((k+1)/bins*width - binSpacing/2)] = point

        if style == "line":
            binSpace = this_height - lineThickness
            paddedFrame = np.full((this_height, width + 2*lineThickness, 3), backgroundColor).astype(np.uint8)

            for k in range(bins):
                vector1Y = int(bins[j,k]*binSpace)

                if k == 0:
                    vector1X = 0
                else:
                    vector1X = int(k/bins*width)
                
                if k == bins - 1:
                    vector2Y = int(bins[j,k]*binSpace)
                    vector2X = frame.shape[1] - 1
                else:
                    vector2Y = int(bins[j,k+1]*binSpace)
                    vector2X = int((k+1)/bins*width)
                
                rr, cc = line(vector1Y, vector1X, vector2Y, vector2X)
                for i in range(len(rr)):
                    paddedFrame[rr[i]:int(rr[i]+lineThickness), int(cc[i] + 0.5*lineThickness):int(cc[i]+ 1.5*lineThickness)] = color
            frame = paddedFrame[:,int(lineThickness):int(-lineThickness)]

        if style == "fill":
            for k in range(bins):
                vector1Y = np.ceil(bins[j,k]*this_height)

                if k == 0:
                    vector1X = 0
                else:
                    vector1X = int(k/bins*width)
                
                if k == bins - 1:
                    vector2Y = np.ceil(bins[j,k]*this_height)
                    vector2X = frame.shape[1] - 1
                else:
                    vector2Y = np.ceil(bins[j,k+1]*this_height)
                    vector2X = int((k+1)/bins*width)

                r = [vector1Y, vector2Y, 0, 0]
                c = [vector1X, vector2X, vector2X, vector1X]
                rr, cc = polygon(r, c, frame.shape)
                frame[rr, cc] = color

    if radial:
        midHeight = int(this_height/2)
        midWidth = int(width/2)
        maxVectorLength = radiusEnd - radiusStart

        if style == "bars":
            for k in range(bins):
                angleStart = k/bins + (binSpacing/2)/width
                angleEnd = (k+1)/bins + (-binSpacing/2)/width

                vector1Y = int(midHeight + radiusStart * radialVectorY(angleStart))
                vector1X = int(midWidth + radiusStart * radialVectorX(angleStart))
                vector2Y = int(midHeight + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorY(angleStart))
                vector2X = int(midWidth + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorX(angleStart))

                vector3Y = int(midHeight + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorY(angleEnd))
                vector3X = int(midWidth + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorX(angleEnd))
                vector4Y = int(midHeight + radiusStart * radialVectorY(angleEnd))
                vector4X = int(midWidth + radiusStart * radialVectorX(angleEnd))

                r = [vector1Y, vector2Y, vector3Y, vector4Y]
                c = [vector1X, vector2X, vector3X, vector4X]
                rr, cc = polygon(r, c, frame.shape)
                frame[rr, cc] = color
        
        if style == "line":
            for k in range(bins):
                angleStart = k/bins
                angleEnd = (k+1)/bins

                vector1Y = int(midHeight + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorY(angleStart) - 1)
                vector1X = int(midWidth + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorX(angleStart) - 1)

                if k == bins - 1:
                    k = k - 1
                
                vector2Y = int(midHeight + (radiusStart + bins[j,k+1]*maxVectorLength) * radialVectorY(angleEnd) - 1)
                vector2X = int(midWidth + (radiusStart + bins[j,k+1]*maxVectorLength) * radialVectorX(angleEnd) - 1)

                rr, cc = line(vector1Y, vector1X, vector2Y, vector2X)

                for i in range(len(rr)):
                    frame[rr[i]:int(rr[i]+lineThickness), int(cc[i] + 0.5*lineThickness):int(cc[i]+ 1.5*lineThickness)] = color
        
        if style == "fill":
            for k in range(bins):
                angleStart = k/bins
                angleEnd = (k+1)/bins

                vector1Y = int(midHeight + radiusStart * radialVectorY(angleStart))
                vector1X = int(midWidth + radiusStart * radialVectorX(angleStart))
                vector2Y = int(midHeight + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorY(angleStart))
                vector2X = int(midWidth + (radiusStart + bins[j,k]*maxVectorLength) * radialVectorX(angleStart))

                if k == bins - 1:
                    k = k - 1
                
                vector3Y = int(midHeight + (radiusStart + bins[j,k+1]*maxVectorLength) * radialVectorY(angleEnd))
                vector3X = int(midWidth + (radiusStart + bins[j,k+1]*maxVectorLength) * radialVectorX(angleEnd))
                vector4Y = int(midHeight + radiusStart * radialVectorY(angleEnd))
                vector4X = int(midWidth + radiusStart * radialVectorX(angleEnd))

                r = [vector1Y, vector2Y, vector3Y, vector4Y]
                c = [vector1X, vector2X, vector3X, vector4X]
                rr, cc = polygon(r, c, frame.shape)
                frame[rr, cc] = color
    
    frame = np.flipud(frame)

    if channel != "stereo":
        if mirror == 1:
            fullFrame[:frame.shape[0],:] = frame
            fullFrame[frame.shape[0]:frame.shape[0]*2,:] = np.flipud(frame)
            frame = fullFrame
        
        elif mirror == 2:
            fullFrame[:frame.shape[0],:] = np.flipud(frame)
            fullFrame[frame.shape[0]:frame.shape[0]*2,:] = frame
            frame = fullFrame
        
        #if image and radial:
            #frame[frameMask] = radialImage[radialImageMask,:3]
    
    return frame

def render_stereo_channel(bins, j):
    height = 540
    width = 1920
    backgroundColor = hex2rgb("000000")
    mirror = 0
    radial = True
    image = None

    left = bins[0]
    right = bins[1]
    frame = np.full((height, width, 3), backgroundColor).astype(np.uint8)

    frame1 = render_mono_channel(left, j)
    frame2 = render_mono_channel(right, j)

    if mirror == 1:
        frame[:frame1.shape[0],:] = frame1
        frame[frame2.shape[0]:frame2.shape[0]*2,:] = np.flipud(frame2)

    elif mirror == 2:
        frame[:frame1.shape[0],:] = np.flipud(frame1)
        frame[frame2.shape[0]:frame.shape[0]*2,:] = frame2

    if radial:
        frame[:,:int(frame1.shape[1]/2)] = np.fliplr(frame1[:,int(frame1.shape[1]/2):])
        frame[:,int(frame2.shape[1]/2):] = frame2[:,int(frame2.shape[1]/2):]

        #if image:
            #frame[frameMask] = radialImage[radialImageMask,:3]
    
    return frame

def renderPoint(bins):
    style = "fill"
    barHeight = -1
    width = 1920
    binWidth = width/bins * (5/6)
    color = hex2rgb("ffffff")
    backgroundColor = hex2rgb("000000")
    if style == "bars":
        point = np.full((int(barHeight), int(binWidth), 3), color)
        return point
    if style == "circles":
        point = np.full((int(binWidth), int(binWidth), 3), backgroundColor)
        rr, cc = disk((int.binWidth/2), (int.binWidth/2), (int.binWidth/2))
        point[rr, cc, :] = color
        return point
    if style == "donuts":
        point = np.full((int(binWidth), int(binWidth), 3), backgroundColor)
        rr, cc = disk((int.binWidth/2), (int.binWidth/2), (int.binWidth/2))
        point[rr, cc, :] = color
        rr, cc = disk((int.binWidth/2), (int.binWidth/2), (int.binWidth/2 - 0.5))
        point[rr, cc, :] = backgroundColor
        return point

def radialVectorY(angle):
    circumference = 180
    rotation = 90
    vector = np.cos(2*np.pi*(angle * circumference + rotation))
    return vector

def radialVectorX(angle):
    circumference = 180
    rotation = 90
    vector = np.sin(2*np.pi*(angle * circumference + rotation))
    return vector

def hex2rgb(hex):
    hexLen = len(hex)
    color = []
    try:
        if hexLen == 6:
            for i in (0, 2, 4):
                color.append(int(hex[i:i+2], 16))
        elif hexLen == 3:
            for i in (0, 1, 2):
                color.append(int(hex[i:i+1]*2, 16))
        else:
            exit("Color is not a valid HEX code, eg. ff0000.")
        return color
    except:
        exit("Color is neither a named color (e.g. red) or a valid HEX code (eg. ff0000).")

def disk(center: tuple, radius, *, shape: tuple | None = None) -> NDArray: ...
def line(r0: int, c0: int, r1: int, c1: int): ...
def polygon(r, c, shape: tuple | None = None) -> NDArray: ...

def print_progress_bar(iteration, total, prefix = "Progress:", suffix = "Complete", decimals = 2, length = 50, fill = '█', printEnd = "\r"):
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
    print(f'\r{prefix} |{bar}| {percent}% ({iteration}/{total}) {suffix}', end = printEnd)

def create_video():
    destination = "visualization"
    processes = cpu_count()
    start = 0
    global end
    filename = "test"

    with open(destination + "/vidList", "x") as vidList:
        for i in range(processes):
            vidList.write("file 'part" + str(i) + VID_EXT + "'\n")
    
    arguments = [
        "ffmpeg",
        '-hide_banner',
        '-loglevel', 'error',
        '-stats',
        '-f', 'concat',
        '-safe', 
        '0',
        '-i', 
        destination + "/vidList",
    ]

    if start != 0:
        arguments += ['-ss', str(start)]
    arguments += ['-i', filename]
    if end != -1:
        arguments += ['-t', str(end - start)]
    
    arguments += [
        '-c:v', 'libx264',
        '-preset', 'ultrafast',
        '-crf', '16',
        '-pix_fmt', 'yuv420p',
        '-c', 'copy',
        '-y', destination+VID_EXT
    ]

    proc = subprocess.Popen(
        arguments,
        stdout=stdout,
        stderr=stderr,
    )

    return proc.wait()

def cleanup_files(directoryExisted):
    destination = "visualization"

    remove(destination + "/vidList")
    for i in range(cpu_count()):
        remove(destination + "/part" + str(i) + VID_EXT)

    if not directoryExisted:
        try:
            rmdir(destination)
        except OSError as error:
            print(error)
            print("Directory " + destination + " could not be removed.")