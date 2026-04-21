dataDir = fullfile(pwd, "CBCTdata");

if ~exist(dataDir, 'dir')
    mkdir(dataDir);
end

% Download the CBCT data from support files
cbctDataFileURL = 'https://ssd.mathworks.com/supportfiles/medical/CBCT_raw_data.zip';
zipFilePath = fullfile(dataDir, 'CBCT_raw_data.zip');
websave(zipFilePath, cbctDataFileURL);

% Unzip the downloaded file into the data directory
unzip(zipFilePath, dataDir);