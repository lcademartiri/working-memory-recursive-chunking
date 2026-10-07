function [sequence,source] = generate_nested_source_v8(cfg,sequenceSeed)
% GENERATE_NESTED_SOURCE_V8
% Fixed nested compositional source with one environmental control R.
%
% The grammar is a depth-D directed acyclic hierarchy.  Every parent is an
% ordered pair of lower-level motifs; lower-level motifs are reused across
% parents.  The top-level stream is built from paired recurrences:
%
%       G ... (R-1 distractor episodes) ... G
%
% so R directly controls an environmental recurrence scale while grammar
% topology and compression gain are held fixed.
%
% The paper should NOT claim that an unmarked recurrence-lag distribution
% alone determines arbitrary environments.  V8 asks how R behaves within a
% fixed recursively compositional ensemble.

if nargin < 2
    sequenceSeed = cfg.sequenceSeed;
end

D = cfg.sourceDepth;
A = cfg.rawAlphabetSize;
K = cfg.motifsPerLevel;

assert(D>=2,'sourceDepth must be >=2.');
assert(A>=K,'rawAlphabetSize should be >= motifsPerLevel.');
assert(cfg.recurrenceLag>=1 && cfg.recurrenceLag==round(cfg.recurrenceLag), ...
    'recurrenceLag must be a positive integer.');

% Grammar is identical across experimental conditions / replicates unless
% grammarSeed is explicitly changed.
rng(cfg.grammarSeed,'twister');

source = struct();
source.depth = D;
source.rawAlphabetSize = A;
source.motifsPerLevel = K;
source.recurrenceLag = cfg.recurrenceLag;
source.motifExpansions = cell(D,1);
source.childPairs = cell(D,1);

% Level 0 "motifs" are atomic symbols 1..A.
prevExp = cell(A,1);
for j=1:A
    prevExp{j} = j;
end
prevCount = A;

for L = 1:D
    % Build K unique pairs with balanced reuse of children.
    p1 = randperm(prevCount);
    if numel(p1) < K
        p1 = repmat(p1,1,ceil(K/numel(p1)));
    end
    p1 = p1(1:K);

    % Find a permutation that gives unique ordered pairs and avoids identity
    % pairs whenever possible.
    success = false;
    for attempt=1:1000
        p2 = randperm(prevCount);
        if numel(p2)<K
            p2 = repmat(p2,1,ceil(K/numel(p2)));
        end
        p2 = p2(1:K);
        pairs = [p1(:),p2(:)];
        if size(unique(pairs,'rows'),1)==K && all(pairs(:,1)~=pairs(:,2))
            success = true;
            break;
        end
    end
    if ~success
        error('Could not construct a unique balanced grammar at level %d.',L);
    end

    currExp = cell(K,1);
    for j=1:K
        currExp{j} = [prevExp{p1(j)}, prevExp{p2(j)}];
    end

    source.childPairs{L} = pairs;
    source.motifExpansions{L} = currExp;
    prevExp = currExp;
    prevCount = K;
end

% Generate top-level episode IDs with explicit paired recurrence lag R.
rng(sequenceSeed,'twister');
R = cfg.recurrenceLag;
topIDs = zeros(cfg.topEpisodes,1);
cursor = 1;

while cursor <= cfg.topEpisodes
    target = randi(K);
    topIDs(cursor) = target;

    % R-1 distractors. Avoid target when possible.
    for g=1:(R-1)
        if cursor+g > cfg.topEpisodes
            break;
        end
        d = randi(K-1);
        if d >= target
            d = d+1;
        end
        topIDs(cursor+g) = d;
    end

    if cursor+R <= cfg.topEpisodes
        topIDs(cursor+R) = target;
    end

    cursor = cursor + R + 1;
end

% Fill any zeros created by truncation edge cases.
z = find(topIDs==0);
topIDs(z) = randi(K,numel(z),1);

% Expand top-level episodes to raw symbols.
topRawLength = numel(source.motifExpansions{D}{1});
sequence = zeros(1,cfg.topEpisodes*topRawLength);
cursor = 1;
for e=1:cfg.topEpisodes
    x = source.motifExpansions{D}{topIDs(e)};
    sequence(cursor:cursor+topRawLength-1) = x;
    cursor = cursor + topRawLength;
end

source.topIDs = topIDs;
source.topRawLength = topRawLength;
source.sequenceLength = numel(sequence);

% Latent aligned motif sequence at every level, useful for source
% recurrence diagnostics.
source.latentLevelIDs = cell(D,1);
for L=1:D
    blockLength = 2^L;
    nBlocks = floor(numel(sequence)/blockLength);
    ids = NaN(nBlocks,1);
    for b=1:nBlocks
        rawBlock = sequence((b-1)*blockLength+1:b*blockLength);
        ids(b) = match_expansion(rawBlock,source.motifExpansions{L});
    end
    source.latentLevelIDs{L} = ids;
end

source.recurrenceStats = source_recurrence_stats(source);

% Map exact raw motif expansion -> source level.
source.exactExpansionKeys = cell(D,1);
for L=1:D
    keys = cell(K,1);
    for j=1:K
        keys{j} = expansion_key(source.motifExpansions{L}{j});
    end
    source.exactExpansionKeys{L} = keys;
end

end


function id = match_expansion(rawBlock,motifCell)
id = NaN;
for j=1:numel(motifCell)
    if isequal(rawBlock,motifCell{j})
        id = j;
        return;
    end
end
end


function stats = source_recurrence_stats(source)
D = source.depth;
rows = repmat(struct('level',0,'nIntervals',0,'meanLagBlocks',NaN, ...
    'medianLagBlocks',NaN,'meanLagRaw',NaN,'medianLagRaw',NaN, ...
    'burstiness',NaN),D,1);

for L=1:D
    ids = source.latentLevelIDs{L};
    allLags = [];
    for id=1:source.motifsPerLevel
        pos = find(ids==id);
        if numel(pos)>=2
            allLags = [allLags; diff(pos(:))]; %#ok<AGROW>
        end
    end
    rows(L).level = L;
    rows(L).nIntervals = numel(allLags);
    if ~isempty(allLags)
        mu = mean(allLags);
        sd = std(allLags);
        rows(L).meanLagBlocks = mu;
        rows(L).medianLagBlocks = median(allLags);
        rows(L).meanLagRaw = mu*2^L;
        rows(L).medianLagRaw = median(allLags)*2^L;
        if mu+sd>0
            rows(L).burstiness = (sd-mu)/(sd+mu);
        end
    end
end
stats = struct2table(rows);
end


function k = expansion_key(x)
k = sprintf('%d,',x);
end
