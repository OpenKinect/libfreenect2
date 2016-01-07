@echo off
echo Stopping tracing...
logman stop -n usbtrace 
logman delete -n usbtrace
timestamp.bat %SystemRoot%\Tracing\usbtrace_000001.etl
echo Tracing completed. See C:\Windows\tracing for the file.