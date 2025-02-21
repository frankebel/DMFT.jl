# IO to read/write data in HDF5 format.

"""
    read_hdf5(filename::Abstractstring, T::Type)

Read data of type `T` from `filename`.

See also [`write_hdf5`](@ref).
"""
function read_hdf5 end

"""
    write_hdf5(filename::Abstractstring, content)

Write `content` to `filename` in HDF5 format.

If the file already exists, it will be overwritten.

See also [`read_hdf5`](@ref).
"""
function write_hdf5 end

# Number
function read_hdf5(filename::AbstractString, T::Type{<:Number})
    h5open(filename, "r") do fid
        scalar::T = read(fid, "s")
        return scalar
    end
end

function write_hdf5(filename::AbstractString, s::Number)
    h5open(filename, "w") do fid
        fid["s"] = s
    end
    return nothing
end

# Array{Number,N}
function read_hdf5(filename::AbstractString, ::Type{<:Array{T,N}}) where {T<:Number,N}
    h5open(filename, "r") do fid
        a::Array{T,N} = read(fid, "a")
        return a
    end
end

function write_hdf5(filename::AbstractString, content::Array{T,N}) where {T,N}
    h5open(filename, "w") do fid
        fid["a"] = content
    end
    return nothing
end

# Vector{Matrix{Number}}
function read_hdf5(
    filename::AbstractString, ::Type{<:Vector{<:Matrix{<:T}}}
) where {T<:Number}
    h5open(filename, "r") do fid
        n::Int = read(fid, "n") # read out length
        result = Vector{Matrix{T}}(undef, n)
        for i in 1:n
            result[i] = read(fid, "$i")
        end
        return result
    end
end

function write_hdf5(filename::AbstractString, content::Vector{Matrix{T}}) where {T<:Number}
    n = length(content)
    h5open(filename, "w") do fid
        fid["n"] = n # store length
        for i in eachindex(content)
            fid["$i"] = content[i] # store each matrix
        end
    end
    return nothing
end

# Pole{A,B}
function read_hdf5(filename::AbstractString, ::Type{<:Pole{A,B}}) where {A,B}
    return h5open(filename, "r") do fid
        a::A = read(fid, "a")
        b::B = read(fid, "b")
        return Pole{A,B}(a, b)
    end
end

function write_hdf5(filename::AbstractString, P::Pole)
    h5open(filename, "w") do fid
        fid["a"] = P.a
        fid["b"] = P.b
    end
    return nothing
end
