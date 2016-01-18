@echo off
echo Initializing USB 3 tracing...
logman create trace -n usbtrace -o %SystemRoot%\Tracing\usbtrace.etl -nb 128 640 -bs 128
logman update trace -n usbtrace -p Microsoft-Windows-USB-USBXHCI (Default,PartialDataBusTrace)
logman update trace -n usbtrace -p Microsoft-Windows-USB-UCX (Default,PartialDataBusTrace)
logman update trace -n usbtrace -p Microsoft-Windows-USB-USBHUB3 (Default,PartialDataBusTrace)
REM uncomment below for usb 2 devices as well (not needed for k4w2)
REM logman update trace -n usbtrace -p Microsoft-Windows-USB-USBPORT
REM logman update trace -n usbtrace -p Microsoft-Windows-USB-USBHUB
logman update trace -n usbtrace -p Microsoft-Windows-Kernel-IoTrace 0 2
logman start -n usbtrace
echo Tracing started. Perform USB tasks now.