@ECHO off
  SETLOCAL
  IF [%1] NEQ [] goto s_start

  :: Author - Simon Sheppard, Nov 2012
  :: Tested for Windows XP, Windows 2008
  Echo StampMe.cmd
  Echo Rename a file with the DATE/Time
  Echo:
  Echo Syntax
  Echo    STAMPME TestFile.txt
  Echo:
  Echo    STAMPME "Test File.txt"
  Echo:
  Echo    STAMPME "c:\docs\Test File.txt"
  Echo:
  Echo    Will rename the file in the format "Test File-2009-12-30@16-55.txt"
  Echo:
  Echo    In a batch file use CALL STAMPME ...
  GOTO :eof
  
  :s_start
  Set _file=%~n1%
  Set _pathname=%~f1%
  Set _ext=%~x1%

  ::Get the date
  For /f "tokens=1-4 delims=/-. " %%G in ('Date /t') Do (Call :s_fixdate %%G %%H %%I %%J)
  Goto :s_time
 
  :s_fixdate
  Set _yr=%1
  if "%_yr:~0,1%" GTR "9" Shift
  For /f "skip=1 tokens=2-4 delims=(-)" %%G in ('Echo.^|Date') Do (
       Set %%G=%1&Set %%H=%2&Set %%I=%3)
  goto :eof

  :s_time
  :: Get the time
  For /f "tokens=1-3 delims=1234567890 " %%a in ("%time%") Do Set "delims=%%a%%b%%c"
  For /f "tokens=1-4 delims=%delims%" %%G in ("%time%") Do (
    Set _hh=%%G
    Set _min=%%H
    Set _ss=%%I
    Set _ms=%%J
  )
  :: Strip any leading spaces
  Set _hh=%_hh: =%

  :: Ensure the hours have a leading zero
  if 1%_hh% LSS 20 Set _hh=0%_hh%
 
REM Echo   Year-Month-Day@Hour-Min-Sec
REM  Echo   %yy%-%mm%-%dd%    @  %_hh%-%_min%-%_ss%
  REN "%_pathname%" "%_file% - %yy%-%mm%-%dd%@%_hh%-%_min%-%_ss%%_ext%"
  Echo File created %_file% - %yy%-%mm%-%dd%@%_hh%-%_min%-%_ss%%_ext%