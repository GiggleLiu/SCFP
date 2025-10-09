using FFTW, Images

# Load and convert image to grayscale
# You can also use `load("myimage.png")` to load the image from your local file system.
img = load(download("https://i.imgur.com/VGPeJ6s.jpg"))
gray_img = Gray.(img)

# Apply 2D FFT
img_data = Float64.(gray_img)  # Convert to Float64
img_fft = fftshift(fft(img_data))

# Visualize frequency spectrum (log scale for better visibility)
spectrum = log.(1 .+ abs.(img_fft))
Gray.(spectrum ./ maximum(spectrum))

# Set threshold - discard coefficients below tolerance
tolerance = 100  # Adjust based on desired compression
img_fft_compressed = copy(img_fft)
img_fft_compressed[abs.(img_fft_compressed) .< tolerance] .= 0

# Convert to sparse matrix for efficient storage
using SparseArrays
sparse_fft = sparse(img_fft_compressed)

# Compression ratio
compression_ratio = nnz(sparse_fft) / length(img_fft)
println("Compression ratio: $(compression_ratio * 100)%")

# Reconstruct image using inverse FFT
img_recovered = ifft(ifftshift(Matrix(sparse_fft)))
recovered_img = Gray.(abs.(img_recovered))


using FFTW

# DCT compression
img_dct = dct(img_data)
img_dct_compressed = copy(img_dct)
tolerance = sort(vec(img_dct_compressed))[round(Int, length(img_dct_compressed) * 0.78)]
img_dct_compressed[abs.(img_dct_compressed) .< tolerance] .= 0

# Reconstruct
img_recovered_dct = idct(img_dct_compressed)
recovered_img_dct = Gray.(abs.(img_recovered_dct))