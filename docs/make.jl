using DMFT
using Documenter

DocMeta.setdocmeta!(DMFT, :DocTestSetup, :(using DMFT); recursive=true)

makedocs(;
    modules=[DMFT],
    authors="Frank Ebel and contributors",
    sitename="DMFT.jl",
    format=Documenter.HTML(;
        canonical="https://frankebel.github.io/DMFT.jl", edit_link="main", assets=String[]
    ),
    pages=["Home" => "index.md"],
)

deploydocs(; repo="github.com/frankebel/DMFT.jl", devbranch="main")
