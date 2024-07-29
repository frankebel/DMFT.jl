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

# only works for public repos, see <https://github.com/frankebel/DMFT.jl/settings/pages>
# deploydocs(; repo="github.com/frankebel/DMFT.jl", devbranch="main")
