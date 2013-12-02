Use these convenience scripts if you want to create USB 3 traces as described in this blog:

http://blogs.msdn.com/b/usbcoreblog/archive/2012/08/07/how-to-trace-usb-3-activity.aspx

Requires Windows 8.1 and software installed as described in the link.

1. Open command prompt as administrator
2. Run starttrace.bat. 
3. When the bat file completes, do something USB-y. Try to keep it short as the trace logs can take up a lot of space. 
4. When you are done, run stoptrace.bat as soon as possible.
5. Check c:\Windows\tracing for the log file, and rename it to describe what you were tracing.

Logs (.etl) are viewable in NetMon on Windows, per the link above.