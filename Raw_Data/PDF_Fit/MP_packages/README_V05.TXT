ManeParse Notes:

------------------------------------
Version 5.0:  April 2021

Improve the parsing of the xxx.info files.

In the xxx.info file, input fields (such as "SetDesc:" ) CANNOT be split across multiple lines. 
While this is valid YAML, ManeParse reads a single line (record) at a time.
The "patch" for this is simply remove \newlines. 
E.g., 
	SetDesc: "This will BREAK \cr
		the ManeParse Program"
	SetDesc: "This will NOT break the ManeParse Program"




------------------------------------
Version 4.0: May 2020 

Add pdfFunctionX[] to return interpolation WITHOUT the "x" factor, i.e., x*f(x,Q), not f(x,Q). This also modifies pdfX[]
