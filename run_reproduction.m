function T = run_reproduction(mode)
% Reproduce the V8.2 synthetic capacity gate and slow-input experiment.
% Usage: run_reproduction('smoke'), 'gate', 'slow', or 'all'.
% MATLAB is required. The two files in src/ are unchanged production code.

if nargin == 0, mode = 'smoke'; end
mode = lower(char(mode));
assert(any(strcmp(mode, {'smoke','gate','slow','all'})), 'Unknown mode.');

root = fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
out = fullfile(root,'results');
if ~exist(out,'dir'), mkdir(out); end

if strcmp(mode,'all')
    A = run_reproduction('gate');
    B = run_reproduction('slow');
    T = [A; B];
    return;
end

cfg.activeCapacity = 6;
cfg.tokenInterval = 1;
cfg.processingTime = 1;
cfg.recursiveReuse = true;
cfg.enforceAssertions = true;
cfg.sourceDepth = 5;
cfg.rawAlphabetSize = 8;
cfg.motifsPerLevel = 8;
cfg.topEpisodes = 360;
cfg.recurrenceLag = 4;
cfg.grammarSeed = 41771;
cfg.sequenceSeed = 9001;
baseSequenceSeed = cfg.sequenceSeed;

switch mode
    case 'smoke'
        nRep = 2;
        capacities = [6 7];
        intervals = 1;
        seedOffset = 10000000;
    case 'gate'
        nRep = 120;
        capacities = 4:10;
        intervals = 1;
        seedOffset = 10000000;
    case 'slow'
        nRep = 50;
        capacities = 4:7;
        intervals = [1 2 4 8 12];
        seedOffset = 11000000;
end

checkpoint = fullfile(out,[mode '.mat']);
if exist(checkpoint,'file')
    S = load(checkpoint,'rows');
    rows = S.rows;
    assert(numel(rows) <= nRep*numel(capacities)*numel(intervals));
else
    rows = repmat(struct('replicate',0,'sequenceSeed',0, ...
        'activeCapacity',0,'tokenInterval',0,'maximumExactSourceLevel',0, ...
        'reachedFullExactDepth',false,'maximumLearnedLevel',0, ...
        'maximumConceptRawSpan',0,'meanContextAmplification',0),0,1);
end

for rep = 1:nRep
    seed = baseSequenceSeed + seedOffset + 100000*rep + 7919*cfg.recurrenceLag;
    cfg.sequenceSeed = seed;
    [sequence,source] = generate_nested_source_v8(cfg,seed);
    for C = capacities
        for dt = intervals
            done = ~isempty(rows) && any([rows.replicate]==rep & ...
                [rows.activeCapacity]==C & [rows.tokenInterval]==dt);
            if done, continue; end
            cfg.activeCapacity = C;
            cfg.tokenInterval = dt;
            R = wm_bootstrap_model_v8(sequence,cfg,source);
            s = R.summary;
            row = struct('replicate',rep,'sequenceSeed',seed, ...
                'activeCapacity',C,'tokenInterval',dt, ...
                'maximumExactSourceLevel',s.maximumExactSourceLevel, ...
                'reachedFullExactDepth',logical(s.reachedFullExactDepth), ...
                'maximumLearnedLevel',s.maximumLearnedLevel, ...
                'maximumConceptRawSpan',s.maximumConceptRawSpan, ...
                'meanContextAmplification',s.meanContextAmplification);
            rows(end+1,1) = row; %#ok<AGROW>
        end
    end
    save(checkpoint,'rows');
    T = sortrows(struct2table(rows),{'replicate','activeCapacity','tokenInterval'});
    writetable(T,fullfile(out,[mode '.csv']));
    fprintf('%s: completed replicate %d/%d\n',mode,rep,nRep);
end

T = sortrows(struct2table(rows),{'replicate','activeCapacity','tokenInterval'});
if ~strcmp(mode,'smoke')
    verify_counts(T,mode,fullfile(root,'expected_counts.csv'));
end
end

function verify_counts(T,mode,path)
expected = readtable(path);
expected = expected(strcmp(expected.experiment,mode),:);
for i = 1:height(expected)
    subset = T(T.activeCapacity==expected.activeCapacity(i) & ...
        T.tokenInterval==expected.tokenInterval(i),:);
    assert(height(subset)==expected.nRuns(i), ...
        'Missing runs for capacity %d, interval %g.', ...
        expected.activeCapacity(i),expected.tokenInterval(i));
    observed = sum(subset.reachedFullExactDepth);
    assert(observed==expected.nReachedLevel5(i), ...
        'Mismatch for capacity %d, interval %g: observed %d, expected %d.', ...
        expected.activeCapacity(i),expected.tokenInterval(i), ...
        observed,expected.nReachedLevel5(i));
end
fprintf('%s: all %d reference count cells matched.\n',mode,height(expected));
end
