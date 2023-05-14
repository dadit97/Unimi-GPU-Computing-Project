%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% University/School Laboratory Report
% LaTeX Class
% Version 4.0 (March 21, 2022)
%
% This class originates from:
% https://www.LaTeXTemplates.com
%
% Authors:
% Vel (vel@latextemplates.com)
% Linux and Unix Users Group at Virginia Tech Wiki 
%
% License:
% CC BY-NC-SA 4.0 (https://creativecommons.org/licenses/by-nc-sa/4.0/)
%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%----------------------------------------------------------------------------------------
%	CLASS CONFIGURATION
%----------------------------------------------------------------------------------------

\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{class}[2022/03/21 Creodocs University/School Laboratory Report Class v4.0]

\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}} % Pass through any extra options specified to the base class
\ProcessOptions\relax % Process class options

\LoadClass{article} % Load the base class

%----------------------------------------------------------------------------------------
%	REQUIRED PACKAGES AND MISC CONFIGURATIONS
%----------------------------------------------------------------------------------------

\usepackage{enumitem} % Required for list customization
\setlist{partopsep=0pt, topsep=0pt} % Customize spacing around and inside lists
\renewcommand{\labelenumi}{\alph{enumi}.} % Change numbering in the enumerate environment by letter rather than number

\setlength{\parindent}{0pt} % Suppress paragraph indentation

\usepackage{graphicx} % Required for including images
\graphicspath{{Figures/}{./}} % Specifies where to look for included images (trailing slash required)

\usepackage{float} % Allows more precisely positioning floats e.g. \begin{figure}[H]

\usepackage{mhchem} % Package for chemical equation typesetting
\usepackage{siunitx} % Provides the \SI{}{} and \si{} commands for typesetting technical/scientific SI units correctly

\usepackage{amsmath} % Required for some math elements 

%----------------------------------------------------------------------------------------
%	MARGINS
%----------------------------------------------------------------------------------------

\usepackage[
	top=4.25cm, % Top margin
	bottom=4cm, % Bottom margin
	left=4.75cm, % Left margin
	right=4.75cm, % Right margin
	%showframe % Uncomment to show frames around the margins for debugging purposes
]{geometry}

%----------------------------------------------------------------------------------------
%	FONTS
%----------------------------------------------------------------------------------------

\usepackage[T1]{fontenc} % Output font encoding for international characters
\usepackage[utf8]{inputenc} % Required for inputting international characters

\usepackage[scale=0.96]{XCharter} % Use the XCharter text font
\usepackage[xcharter]{newtxmath} % Set the math font

%----------------------------------------------------------------------------------------
%	BIBLIOGRAPHY
%----------------------------------------------------------------------------------------

\usepackage[
	backend=biber, % Use the biber backend for compiling the bibliography
	style=numeric,
	sorting=ynt
]{biblatex}