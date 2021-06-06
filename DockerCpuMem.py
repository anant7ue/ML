
#!/usr/bin/python3
import os
import sys
import time
from time import sleep
import subprocess

##################################################################

def getRespTime(fname):
    outFile = open(fname, "r")
    respTimeOp = ['-1']
    for line in outFile.readlines():
        if final in line:
            metrics = line.split()
            respTimeOp = metrics[len(metrics)-1].split(respUnit)
            thputIops = metrics[5]
            thputMbps = metrics[9].split('m')
            #print line, respTimeOp[0] (thputMbps[0]) (metrics) (thputIops)
    return thputIops + ",   " + thputMbps[0]+ ",  "+ respTimeOp[0]

##################################################################

def getFileStr(count):
    return f"        flowop openfile name=openfile{count},filesetname=bigfileset,fd=1 \n\
        flowop readwholefile name=readfile{count},fd=1,iosize=$iosize      \n\
        flowop closefile name=closefile{count},fd=1 \n"

def getParamStr(cpu,mem,numFiles,ioSize,numThread,memThread, sep):
    return str(cpu)+ sep +str(mem)+ sep + str(numFiles) + sep + str(ioSize) + sep + str(numThread) + sep + str(memThread)

##################################################################

mc = len(sys.argv)
usageStr = "Usage: script.py 100(cpu%) 100(mem in MB) \n"
if(mc == 1):
    print(usageStr)
    mArgs = 110
    cArgs = 100
elif(mc == 2):
    print(usageStr)
    cArgs = int(sys.argv[1])
    mArgs = 110
elif(mc == 3):
    cArgs = int(sys.argv[1])
    if(cArgs != 100):
        print(usageStr)
        exit(0)
    mArgs = int(sys.argv[2])
else:
    print(usageStr)
    exit(0)

contName = "null"
mapMemToHost = {
         60: "distracted_neumann" ,
         70: "zen_leavitt" ,
         80: "agitated_jepsen" ,
         90: "xenodochial_shirley" ,
         100: "youthful_driscoll" ,
         110: "nice_mclaren" ,
         120: "crazy_golick" ,
         130: "stupefied_wilson" ,
         140: "clever_nobel"
         }

mapCpuToHost = {
        50: "epic_elgamal",
        100: "nice_mclaren",
        150: "clever_nobel",
        200: "unruffled_herschel",
        250: "bold_swanson",
        300: "pensive_chatelet",
        }

if(cArgs == 100):
    contName = mapMemToHost.get(mArgs)
elif(mArgs == 110):
    contName = mapCpuToHost.get(cArgs)

#os.system("docker container lscp fbserver.f "+contName+":/opt/filebench/workloads/")
contRunning = 0
chkOutput = subprocess.check_output("docker container ls > .docker.container", shell=True)
chkCont = open(".docker.container", "r")
for line in chkCont:
    if contName in line:
        contRunning = 1

if(contRunning == 0):
    print("Container with given Cpu Memory config not running. Exiting ")
    exit(0)

dataFile = "trainingData-c" + str(cArgs) + "-m" + str(mArgs)+".csv"
print("Cpu = " + str(cArgs) + " Memory= " + str(mArgs) + "  container_name= " + contName + " Output_file= " + dataFile + '\n')

header='\nset $dir=/tmp  \n\
set $meandirwidth=20   \n\
set $meanappendsize=16k \n'

runTime = 10
newline='\n'
fsStr='set $filesize='
nfStr='set $nfiles='
numtStr='set $nthreads='
ioszStr='set $iosize='
memStr='set $memthread='

procHeadStr='\ndefine fileset name=bigfileset,path=$dir,size=$filesize,entries=$nfiles,dirwidth=$meandirwidth,prealloc=100,readonly \n\
define fileset name=logfiles,path=$dir,size=$filesize,entries=1,dirwidth=$meandirwidth,prealloc \n\
 \n\
define process name=filereader,instances=1 \n\
{ \n\
  thread name=filereaderthread,memsize=$memthread,instances=$nthreads \n\
  { \n\
'
procTailStr='   } \n\
} \n \n\
echo \"Web-server Version 3.1 personality successfully loaded\" \n \n\
run ' + str(runTime)

##################################################################

filesize='100m'
filesize = 'cvar(type=cvar-gamma,parameters=mean:16384;gamma:1.5) '

nfiles = 800
numFileStep = int(nfiles / 20)
fileStepCount = 1

iosize = 1
ioStepCount = 1

nthreads = 50
nThreadStep = int(nthreads / 5)
nThreadStepCount = 1

memThread = 4
memThreadStep = int(memThread / 2)
memThreadStepCount = 1

final = 'Summary'
respUnit = 'ms'

##################################################################

nfiles = 960
# docker container should be running

trainData = open(dataFile, "w")
trainData.write( "Cpu   Memory(MB)   NumFiles   IOSize(MB)   NumThreads   ThreadMemory(MB)   Thput(Iops/sec)   Thput(Mbps)   RespTime(ms/Op) \n")

for nf in range(fileStepCount):
    print("Setting num files to " + str(nfiles) + '\n')
    iosize=5
    for io in range(ioStepCount):
        print("  Setting ioSize to (MB) " + str(iosize) + '\n')
        nthreads=90
        for nt in range(nThreadStepCount):
            print("    Setting number of Threads to " + str(nthreads))
            memThread=12
            for memt in range(memThreadStepCount):
                print("      Setting thread memory to (MB) " + str(memThread))

                cfgFile = open("fbserver.f", "w")
                cfgFile.write(header)
                cfgFile.write(fsStr+str(filesize)+ newline)
                cfgFile.write(nfStr+str(nfiles)+ newline)
                cfgFile.write(ioszStr+str(iosize)+ 'm'+newline)
                cfgFile.write(numtStr+str(nthreads)+ newline)
                cfgFile.write(memStr+str(memThread)+ 'm'+newline)
                cfgFile.write(procHeadStr + newline)
                for i in range(10):
                    #print(getFileStr(i+1))
                    cfgFile.write(getFileStr(i+1))
                cfgFile.write(procTailStr + newline)
                cfgFile.close()
                # copy file to docker
                print("          docker cp fbserver.f "+contName+":/opt/filebench/workloads/")
                t1 = time.time()
                os.system("docker cp fbserver.f "+contName+":/opt/filebench/workloads/")
                logFile = "resp-"+ getParamStr(cArgs, mArgs, nfiles, iosize, nthreads, memThread, "-")
                outputFull = subprocess.check_output("docker exec -it " + contName + " filebench -f /opt/filebench/workloads/fbserver.f | tee sampleOut > " + logFile, shell=True)
                t2 = time.time()
                # run file parse output
                print("        Execution complete in " + str(t2-t1) + "sec; Now writing results ")
                expTime = round(t2-t1,2)
                respTime = getRespTime('sampleOut')
                trainData.write( getParamStr(cArgs, mArgs, nfiles, iosize, nthreads, memThread, ",       ") + ", " + str(expTime) + ", " + respTime + '\n')

                memThread += memThreadStep
            nthreads += nThreadStep
        iosize += 1
    nfiles += numFileStep

trainData.close()

##################################################################
##################################################################
