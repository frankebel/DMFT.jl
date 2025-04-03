using Changelog
using DMFT
using Documenter

DocMeta.setdocmeta!(DMFT, :DocTestSetup, :(using DMFT); recursive=true)

# generate changelog
Changelog.generate(
    Changelog.Documenter(),
    joinpath(@__DIR__, "../CHANGELOG.md"),
    joinpath(@__DIR__, "src/changelog.md");
    repo="frankebel/DMFT.jl",
)

makedocs(;
    modules=[DMFT],
    authors="Frank Ebel and contributors",
    sitename="DMFT.jl",
    format=Documenter.HTML(;
        canonical="https://frankebel.github.io/DMFT.jl", edit_link="main", assets=String[]
    ),
    pages=["Home" => "index.md", "Changelog" => "changelog.md"],
)

# only works for public repos, see <https://github.com/frankebel/DMFT.jl/settings/pages>
# deploydocs(; repo="github.com/frankebel/DMFT.jl", devbranch="main")
