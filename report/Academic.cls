%% Academic.cls — University of Birmingham MSc Project Report Class
%% Reconstructed from template structure
\NeedsTeXFormat{LaTeX2e}
\ProvidesClass{Academic}[2026/07/15 UoB MSc Advanced Project Report]

\LoadClass[12pt,a4paper]{article}

%% ── Packages ──
\RequirePackage[utf8]{inputenc}
\RequirePackage[T1]{fontenc}
\RequirePackage{mathptmx}            % Times Roman
\RequirePackage[top=25mm,bottom=25mm,left=25mm,right=25mm]{geometry}
\RequirePackage{setspace}
\RequirePackage{graphicx}
\RequirePackage{booktabs}
\RequirePackage{caption}
\captionsetup[table]{position=below}
\RequirePackage{subcaption}
\RequirePackage{float}
\RequirePackage{amsmath,amssymb}
\RequirePackage{siunitx}
\RequirePackage{hyperref}
\RequirePackage{cleveref}
\RequirePackage{enumitem}
\RequirePackage{tabularx}
\RequirePackage{multirow}
\RequirePackage{xcolor}
\RequirePackage{appendix}
\RequirePackage{pdflscape}
\RequirePackage[UKenglish]{babel}

%% ── Bibliography (IMechE numeric style) ──
\RequirePackage[backend=biber,
  style=numeric-comp,
  sorting=none,
  maxbibnames=6,
  minbibnames=3,
  giveninits=true,
  doi=true,
  url=false,
  isbn=false]{biblatex}

%% ── Customisable fields ──
\newcommand{\myTitle}{Title of Paper Here}
\newcommand{\MyAuthor}{Your Name Here}
\newcommand{\MyDepartment}{Department Here}
\newcommand{\ID}{1234567}
\newcommand{\Keywords}{Keyword 1, Keyword 2}
\newcommand{\MySupervisor}{Supervisor's Name}

%% ── Abstract storage ──
\newcommand{\myabstract}[1]{\long\gdef\@abstractcontent{#1}}
\myabstract{}

%% ── Title page (UoB style) ──
\renewcommand{\maketitle}{%
  \begin{titlepage}
    \centering
    \vspace*{0.5cm}
    \hspace*{-3cm}\includegraphics[width=0.75\textwidth]{figures/uob_logo.png}\par
    \vspace{0.8cm}
    {\large College of Engineering and Physical Sciences\par}
    {\large \MyDepartment\par}
    \vspace{0.4cm}
    {\large MSc Robotics\par}
    {\large Advanced Project --- 2025--26 Session\par}
    \vspace{0.6cm}
    {\LARGE\bfseries FINAL REPORT\par}
    \vspace{0.8cm}
    {\LARGE\bfseries \myTitle\par}
    \vspace{1.2cm}
    {\Large \MyAuthor\par}
    \vspace{0.3cm}
    {\large Student ID: \ID\par}
    \vspace{0.3cm}
    {\large Supervisor: \MySupervisor\par}
    \vspace{0.8cm}
    {\large \today\par}
    \vfill
  \end{titlepage}
  %% Abstract page
  \newpage
  \begin{center}
    {\Large\bfseries Abstract\par}
  \end{center}
  \vspace{0.5em}
  \@abstractcontent
  \vspace{1em}
  \par\noindent\textbf{Keywords:} \Keywords
  \newpage
}

%% ── Section formatting ──
\renewcommand{\thesection}{\arabic{section}}
\renewcommand{\thesubsection}{\thesection.\arabic{subsection}}

%% ── Caption formatting ──
\captionsetup[table]{position=above, skip=4pt}
\captionsetup[figure]{position=below, skip=4pt}
\captionsetup{font=small, labelfont=bf}

%% ── Hyperref setup ──
\hypersetup{
  colorlinks=true,
  linkcolor=black,
  citecolor=blue!60!black,
  urlcolor=blue!60!black
}

\endinput
