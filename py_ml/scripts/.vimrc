" check one time after 4s of inactivity in normal mode
set autoread                                                                                                                                                                                    
" All of your Plugins must be added before the following line
filetype plugin indent on    " required

set cindent
set cino+=t0
set cinoptions+=g-1
"set cino=j1
"
set t_BE=
set autoread
set ignorecase
set fileformat=unix
set shiftwidth=4
set ts=4
"set tw=80
set expandtab
"set list
set visualbell
set ruler
set tags=./tags,tags;$HOME
set nu
syntax on
set wildmenu
set incsearch
set hlsearch
"set fileencodings=utf8,gb18030,big5,gb2312
setlocal indentexpr=
set backspace=indent,eol,start
:nnoremap <Leader>s :%s/\<<C-r><C-w>\>//g<Left><Left>
:nnoremap <C-n> :bnext<CR>
filetype plugin on
set nocp
set noic
"set guifont=-*-courier-medium-r-normal-*-*-120-*-*-m-*-*-*
set guifont=<Tab>
"folding settings
set foldmethod=marker
set foldnestmax=10
set nofoldenable
set foldlevel=1
"mouse
"set mouse=a
map <ScrollWheelUp> <C-Y>
map <ScrollWheelDown> <C-E>
map <C-c> "+y <CR>
set nowrap
set clipboard=unnamed
"auto fill a line with *
func Addtitle()
                call append(line("."), "//***********************************************************************************")
                call append(line(".")+1, "//    ")
                let nline = line(".")
                call append(line(".")+2, "//***********************************************************************************")
                call cursor(nline+2,4)
                exec "startinsert"
endfunc
map <F3> : call Addtitle() <CR>
func Add_intro()
                call append(line("."), "//----------------------------------------------------------------------------------------")
                call append(line(".")+1, "//    ")
                let nline = line(".")
                call append(line(".")+2, "//----------------------------------------------------------------------------------------")
                call cursor(nline+2,4)
                exec "startinsert"
endfunc
map <F4> : call Add_intro() <CR>
func Add_head()
    call append(0,"//*************************************************************************************//")
    call append(1, "//                         Copyright TTI-TSMC, 2015-2018                               //")
    let nline = line(".")
    call append(2, "//*************************************************************************************//")
    call cursor(4,1)
endfunc
map <F2> : call Add_head() <CR>
map <F4> : execute "normal! i" . "#"( line(".") )" " <CR>
nnoremap  dl  "_dd 
noremap! <M-j> <Down>
noremap! <M-k> <Up>
noremap! <M-h> <Left>
noremap! <M-l> <Right>
inoremap \fn <C-R>=expand("%:t:")<CR>
"automatically set screen title
set title
set titlestring=\ %-25.55F\ %a%r%m titlelen=70
"====================================================================
"----- for cvim ------
let g:C_UseTool_cmake = 'yes'
let g:C_UseTool_doxygen = 'yes'
"================ Clang_Complete ===============
"let g:clang_library_path='/tools/software/llvm/3.8.0/lib/libclang.so'
"let g:clang_library_path='/tools/software/sys2/lib'
let g:clang_library_path='/tools/lib'
let g:clang_use_library=1
let g:clang_memory_percent=90
let g:clang_complete_copen=0
let g:clang_complete_auto=1
let g:clang_complete_macros=1
let g:clang_complete_patterns=1
let g:clang_auto_select=0
let g:clang_hl_errors=1
let g:clang_debug=1
let g:clang_periodic_quickfix=0
"let g:clang_user_options='|| exit 0'
let g:clang_user_options="-I/tools/include -std=c++14 || exit 0"
"let g:clang_user_options="-std=c++11 || exit 0"
set conceallevel=2
set concealcursor=vin
let g:clang_snippets=1
let g:clang_conceal_snippets=1
let g:clang_snippets_engine='clang_complete'
set completeopt=menu,menuone
set pumheight=20
function F5_action()
    :edit!
    :call g:ClangUpdateQuickFix()
    ":only
    ":TlistToggle
endfunction
nmap <F5> : let l=line(".") \| let c=line(".") \| exec F5_action() \| call cursor(l,c)<CR>
"=============== TagList =================
let Tlist_Use_Right_Window=1
let Tlist_File_Fold_Auto_Close=1
let Tlist_Sort_Type="name"
"=============== supertab ================
let g:SuperTabRetainCompletionType=2
let g:SuperTabDefaultCompletionType='<c-x><c-u><c-p>'
let g:SuperTabDefaultCompletionType="<c-x><c-o>"
"=============== spell ===================
"setlocal spell spelllang=en_us

"modify the last modified time
function SetLastModifiedTime(lineno)

    let modif_time = strftime( '%m/%d/%Y %I:%M:%S %p', getftime(bufname('%')) )
    if a:lineno == "-1"
        let line = getline(11)
    else
        let line = getline(a:lineno)
    endif
    if line =~ '^////\sLast Modified'
        let line = substitute(line, ':\s\+.*\d\{2\}', ': " .modif_time, "")
    else
        let line = '//  Last Modified:  ' .modif_time ."  by Fangbo Xu (xfb)"
    endif
    if a:lineno == "-1"
        call setline(11,line)
    else
        call append(a:lineno, line)
    endif
endfunction
function! ResCur()
    if line("'\") <= line("$")
    normal! g`"
    return 1
    endif
endfunction
function AddSpaceBinaryOperator()
   let l = line(".")
   let c = col(".")

   " << 
   silent! %s/\(^ *\)\@<! \= *<< */ << /g
   silent! %s/<< \+=/<<=/g

   " <
   silent!  g/^ *\<for\>\|\^ *\<if\>\|^ *\<while\>/s/<=\@!/ < /g 
   silent!  g/^ *\<if\>/s/<\(=\|\)\@!/ < /g
   silent!  g/^ *\<if\>/s/ \+< \+/ < /g
   silent!  g/^ *\<for\>/s/ \+< \+/ < /g
   silent!  g/^ *\<while\>/s/ \+< \+/ < /g
   silent!  g/\<template\>/s/< */</g
   silent!  g/\<template\>/s/ *> *$/>/g
   " >
   silent!  g/^ *\<for\>\|\^ *\<if\>\|^ *\<while\>/s/ *>=\@! */ > /g 
   silent!  g/^ *\<for\>\|\^ *\<if\>\|^ *\<while\>/s/- \+> \+/->/g 
   " ==
   silent!  g/^ *\<if\>/s/ *== */ == /g
   " >> <<
   silent!  %s/ *< \+< */ << /g
   "silent!  %s/ *> \+> */ >> /g

   "=
   silent!  g/^ *\<for\>/s/\>=\</ = /g 
   silent!  %s/\>=\</ = /g
   silent!  %s/\>= *\</ = /g

   "for( and if(
   silent!  g/^ *\<for\>\|\^ *\<if\>/s/for *( */for ( /g 
   silent!  g/^ *\<for\>/s/ *) *$/ )/g 
   silent!  g/^ *\<if\>/s/if *( */if ( /g 
   silent!  g/^ *\<if\>/s/ *) *$/ )/g 
   silent!  g/^ *\<for\>/s/ *) *{$/ ) {/g 
   silent!  g/^ *\<if\>/s/ *) *{$/ ) {/g 
   silent!  g/^ *\<while\>/s/ *) *{$/ ) {/g 
   silent!  g/^ *\<while\>/s/while *( */while ( /g 

   " ,
   silent! %s/\("\|'.*\)\@<! *, */, /g

   " ;
   silent!  g!/}.*$\|; \+\/\/\|"/s/;/ ; /g
   silent!  g!/}.*$\|; \+\/\/\|"/s/ \+; \+/; /g

   " {
   silent!  %s/{ \+/{ /g

   ";}
   silent!  %s/{\@<!;}/; }/g

   " //
"   silent %s/^ *\/\/\*/\/\/\*/g

   "trim space at line end
   silent %s/ \+$//g

   "remove empty lines before closing }
   silent! %s/^\n\( *}\)\@=//g
   silent! %s/\({ *\n\)\@<= *\n//g

   call cursor(l,c)
endfunction
function RemoveDoubleBlankLines()
    let l = line(".")
    let c = col(".")
    " remove all trailing whitespaces of each line
    %s/\s\+$//e
    " merge multiple blank lines into a single line
    silent! g/^$/,/./-j
    " merge blank line preceded by bracket 
    silent! g/{$/,/./-j

    call cursor(l,c)
    
    call AddSpaceBinaryOperator()
endfunction
"au BufWrite *.h call SetLastModifiedTime(-1)
"au BufWrite *.cpp call SetLastModifiedTime(-1)
au BufWrite *.cpp call RemoveDoubleBlankLines()
au BufWrite *.h   call RemoveDoubleBlankLines()
au BufWrite *.py   call RemoveDoubleBlankLines()

" Right alignment of visual block
function! RightAlignVisual() range
let lim = [virtcol("'<"), virtcol("'>")]
let [l, r] = [min(lim), max(lim)]
exe "'<,'>" 's/\%'.l.'v.*\%<'.(r+1).'v./\=StrPadLeft(submatch(0),r-l+1)'
endfunction
function! StrPadLeft(s, w)
let s = substitute(a:s, '^\s\+\|\s\+$', '', 'g')
return repeat(' ', a:w - strwidth(s)) . s
endfunction

vmap <leader>raa : call RightAlignVisual() <CR>
"if exists('+colorcolumn')
    "set colorcolumn=80
"else
    "au BufWinEnter * let w:m2=matchadd('ErrorMsg','\%80v.\+',-1)
"endif

