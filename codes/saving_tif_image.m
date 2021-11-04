% saving a matrix of doubles as tiff image
filename = 'testtifffp.tiff';
A = mCorr;
t = Tiff(filename, 'w');
tagstruct.ImageLength = size(A, 1);
tagstruct.ImageWidth = size(A, 2);
tagstruct.Compression = Tiff.Compression.None;
tagstruct.SampleFormat = Tiff.SampleFormat.IEEEFP;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.BitsPerSample = 64;
tagstruct.SamplesPerPixel = size(A,3);
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
t.setTag(tagstruct);
t.write(A);
t.close();

