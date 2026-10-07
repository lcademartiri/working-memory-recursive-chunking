function R = wm_bootstrap_model_v8(sequence,cfg,source,initialState)
% WM_BOOTSTRAP_MODEL_V8
% Patch level: V8.2 fixes MATLAB 1x0-struct append phantom jobs.
%
% Minimal active-workspace-gated recursive-compression model.
%
% There are NO:
%   candidate registries
%   significance thresholds
%   concept-store budgets
%   forgetting constants
%   strengthening gains
%   variable motif lengths
%   maturation gates
%   ACT-R retrieval parameters
%
% The only internal time is tau = processingTime = 1, defining the unit.
%
% RULES
% 1. Raw symbols arrive every dt into a C-slot active workspace.
% 2. An UNKNOWN adjacent pair can become a concept only if TWO
%    non-overlapping copies are simultaneously present.
% 3. Those four supporting active items must remain valid for tau.
%    If they are evicted/substituted first, the candidate disappears and
%    leaves NO persistent trace.
% 4. A learned concept can compress a matching adjacent pair after tau,
%    provided the pair remains available.
% 5. Learning a concept immediately recodes the two supporting copies.
% 6. Learned concept tokens may themselves participate in new pair
%    induction iff recursiveReuse=true.
%
% This implements the closed loop:
%
% active horizon -> learnable recurrence -> compression -> expanded active
% horizon -> newly learnable recurrence.
%
% initialState is optional and is used for the pretrained/hysteresis branch.
% The active workspace is always empty at the start; only confirmed concepts
% are inherited.

if nargin < 3 || isempty(source)
    source = struct();
end
if nargin < 4
    initialState = [];
end

assert(abs(cfg.processingTime-1.0)<1e-12, ...
    'V8 fixes processingTime=1 as the time unit; do not sweep it.');
assert(cfg.activeCapacity>=4);
assert(cfg.tokenInterval>0);

rawMaxToken = max(sequence);

% ----------------------------- Buffer ----------------------------------
B = empty_buffer();
nextUid = 1;

% ----------------------------- Concepts --------------------------------
if isempty(initialState)
    concepts = repmat(empty_concept(),0,1);
    nextConceptToken = rawMaxToken + 1;
else
    concepts = initialState.concepts;
    if isempty(concepts)
        nextConceptToken = rawMaxToken+1;
    else
        nextConceptToken = max([concepts.token])+1;
        for k=1:numel(concepts)
            concepts(k).inherited = true;
        end
    end
end

pairMap = containers.Map('KeyType','char','ValueType','double');
for k=1:numel(concepts)
    pairMap(pair_key(concepts(k).child1,concepts(k).child2)) = k;
end
initialConceptCount = numel(concepts);

% Only ACTIVE processing jobs are retained. Failed candidate work is deleted.
jobs = reshape(repmat(empty_job(),0,1),[],1);
events = repmat(empty_event(),0,1);
nextEventIndex = 1;

nRaw = numel(sequence);
arrivalTimes = (0:nRaw-1)*cfg.tokenInterval;

% ----------------------------- History ---------------------------------
H.tokenIndex = (1:nRaw)';
H.time = arrivalTimes(:);
H.bufferSlots = zeros(nRaw,1);
H.effectiveRawSpan = zeros(nRaw,1);
H.contextAmplification = zeros(nRaw,1);
H.conceptCount = zeros(nRaw,1);
H.newConceptCount = zeros(nRaw,1);
H.maxLearnedLevel = zeros(nRaw,1);
H.maxExactSourceLevel = zeros(nRaw,1);
H.maxConceptRawSpan = ones(nRaw,1);
H.learnSuccesses = zeros(nRaw,1);
H.compressSuccesses = zeros(nRaw,1);
H.failedJobs = zeros(nRaw,1);

learnSuccesses = 0;
compressSuccesses = 0;
failedJobs = 0;

for rawIndex = 1:nRaw
    t = arrivalTimes(rawIndex);

    [B,concepts,pairMap,jobs,events,nextConceptToken,nextUid, ...
        learnSuccesses,compressSuccesses,failedJobs,nextEventIndex] = ...
        process_until(t,B,concepts,pairMap,jobs,events,nextConceptToken, ...
        nextUid,learnSuccesses,compressSuccesses,failedJobs, ...
        nextEventIndex,cfg,source,rawIndex);

    % Make room for the next raw symbol.
    while numel(B.token) >= cfg.activeCapacity
        doomedUid = B.uid(1);
        [jobs,events,concepts,failedJobs] = invalidate_jobs( ...
            jobs,events,concepts,doomedUid,t,'evicted',failedJobs);
        B = remove_buffer_item(B,1);
    end

    % Append raw symbol.
    B.token(end+1,1) = sequence(rawIndex);
    B.uid(end+1,1) = nextUid;
    B.rawSpan(end+1,1) = 1;
    B.level(end+1,1) = 0;
    B.rawStart(end+1,1) = rawIndex;
    B.rawEnd(end+1,1) = rawIndex;
    nextUid = nextUid + 1;

    [jobs,events,nextEventIndex] = scan_and_schedule( ...
        B,concepts,pairMap,jobs,events,nextEventIndex,t,rawIndex,cfg);

    % History after arrival + scheduling.
    H.bufferSlots(rawIndex) = numel(B.token);
    H.effectiveRawSpan(rawIndex) = sum(B.rawSpan);
    if ~isempty(B.token)
        H.contextAmplification(rawIndex) = ...
            sum(B.rawSpan)/numel(B.token);
    else
        H.contextAmplification(rawIndex) = 1;
    end

    H.conceptCount(rawIndex) = numel(concepts);
    H.newConceptCount(rawIndex) = numel(concepts)-initialConceptCount;
    if isempty(concepts)
        H.maxLearnedLevel(rawIndex) = 0;
        H.maxExactSourceLevel(rawIndex) = 0;
        H.maxConceptRawSpan(rawIndex) = 1;
    else
        H.maxLearnedLevel(rawIndex) = max([concepts.level]);
        ex = [concepts.exactSourceLevel];
        H.maxExactSourceLevel(rawIndex) = max([0 ex]);
        H.maxConceptRawSpan(rawIndex) = max([concepts.rawSpan]);
    end

    H.learnSuccesses(rawIndex) = learnSuccesses;
    H.compressSuccesses(rawIndex) = compressSuccesses;
    H.failedJobs(rawIndex) = failedJobs;

    if cfg.enforceAssertions
        assert(numel(B.token)<=cfg.activeCapacity);
        assert(all(B.rawSpan>=1));
    end
end

% End-of-stream pending jobs are unfinished; they did not learn.
for j=1:numel(jobs)
    ei = jobs(j).eventIndex;
    if strcmp(events(ei).outcome,'pending')
        events(ei).outcome = 'failure';
        events(ei).failureReason = 'unfinished_end_of_stream';
        events(ei).completionTime = NaN;
        validate_job_scalar_fields(jobs(j),j);
        if isequal(jobs(j).typeCode,1) && jobs(j).conceptIndex>0
            concepts(jobs(j).conceptIndex).failureCount = ...
                concepts(jobs(j).conceptIndex).failureCount + 1;
        end
        failedJobs = failedJobs + 1;
    end
end
jobs = reshape(repmat(empty_job(),0,1),[],1);

history = struct2table(H);
eventTable = events_to_table(events);

summary = summarize(concepts,history,eventTable,initialConceptCount, ...
    learnSuccesses,compressSuccesses,failedJobs,cfg,source);

R = struct();
R.config = cfg;
R.summary = summary;
R.history = history;
R.events = eventTable;
R.concepts = concepts;
R.finalState = struct('concepts',concepts);
R.initialConceptCount = initialConceptCount;

if cfg.enforceAssertions
    validate_run(R,cfg);
end
end


% ======================================================================
% EVENT LOOP
% ======================================================================

function [B,concepts,pairMap,jobs,events,nextConceptToken,nextUid, ...
    learnSuccesses,compressSuccesses,failedJobs,nextEventIndex] = ...
    process_until(limitTime,B,concepts,pairMap,jobs,events, ...
    nextConceptToken,nextUid,learnSuccesses,compressSuccesses,failedJobs, ...
    nextEventIndex,cfg,source,rawIndex)

tol = 1e-12;

while ~isempty(jobs)
    completionTimes = [jobs.completionTime];
    [nextTime,jidx] = min(completionTimes);
    if nextTime > limitTime + tol
        break;
    end

    job = jobs(jidx);
    jobs(jidx) = [];
    jobs = jobs(:);  % force canonical 0x1 / Nx1 orientation after deletion
    ei = job.eventIndex;

    validate_job_scalar_fields(job,jidx);
    if isequal(job.typeCode,1)
        % ------------------- Learned concept recognition ----------------
        [ok,pos] = locate_pair(B,job.uids(1),job.uids(2),job.pair);
        if ~ok
            events(ei).outcome = 'failure';
            events(ei).failureReason = 'invalidated_before_compression';
            events(ei).completionTime = nextTime;
            concepts(job.conceptIndex).failureCount = ...
                concepts(job.conceptIndex).failureCount + 1;
            failedJobs = failedJobs + 1;
            continue;
        end

        [B,jobs,events,concepts,failedJobs,nextUid] = ...
            substitute_pair(B,pos,job.conceptIndex,job.eventIndex, ...
            jobs,events,concepts,failedJobs,nextUid,nextTime);

        concepts(job.conceptIndex).useCount = ...
            concepts(job.conceptIndex).useCount + 1;
        compressSuccesses = compressSuccesses + 1;

        events(ei).outcome = 'success';
        events(ei).completionTime = nextTime;
        events(ei).resultLevel = concepts(job.conceptIndex).level;
        events(ei).resultRawSpan = concepts(job.conceptIndex).rawSpan;

    else
        % ------------------- New concept induction ----------------------
        key = pair_key(job.pair(1),job.pair(2));

        % If another event somehow learned it already, this job is redundant.
        if isKey(pairMap,key)
            events(ei).outcome = 'failure';
            events(ei).failureReason = 'pair_already_learned';
            events(ei).completionTime = nextTime;
            failedJobs = failedJobs + 1;
            continue;
        end

        [ok1,pos1] = locate_pair(B,job.uids(1),job.uids(2),job.pair);
        [ok2,pos2] = locate_pair(B,job.uids(3),job.uids(4),job.pair);
        if ~(ok1 && ok2)
            events(ei).outcome = 'failure';
            events(ei).failureReason = 'evidence_left_active_workspace';
            events(ei).completionTime = nextTime;
            failedJobs = failedJobs + 1;
            continue;
        end

        % Create the concept.
        c = empty_concept();
        c.index = numel(concepts)+1;
        c.token = nextConceptToken;
        c.child1 = job.pair(1);
        c.child2 = job.pair(2);
        c.level = 1 + max(token_level(job.pair(1),concepts), ...
                          token_level(job.pair(2),concepts));
        c.rawSpan = token_raw_span(job.pair(1),concepts) + ...
                    token_raw_span(job.pair(2),concepts);
        c.formationTime = nextTime;
        c.formationTokenIndex = rawIndex;
        c.useCount = 0;
        c.failureCount = 0;
        c.inherited = false;
        c.rawExpansion = combined_expansion(job.pair,concepts,source);
        c.exactSourceLevel = exact_source_level(c.rawExpansion,source);

        concepts(end+1,1) = c; %#ok<AGROW>
        pairMap(key) = c.index;
        nextConceptToken = nextConceptToken + 1;
        learnSuccesses = learnSuccesses + 1;

        events(ei).outcome = 'success';
        events(ei).completionTime = nextTime;
        events(ei).conceptIndex = c.index;
        events(ei).resultLevel = c.level;
        events(ei).resultRawSpan = c.rawSpan;
        events(ei).exactSourceLevel = c.exactSourceLevel;

        % Learning the representation includes recoding the two supporting
        % copies. Compress rightmost first so the left index remains valid.
        pairPositions = sort([pos1 pos2],'descend');
        pairUids = {job.uids(1:2),job.uids(3:4)};

        for q=1:numel(pairPositions)
            % Re-find by UID because the first substitution changes positions.
            thisUids = pairUids{q};
            [ok,pos] = locate_pair(B,thisUids(1),thisUids(2),job.pair);
            if ok
                [B,jobs,events,concepts,failedJobs,nextUid] = ...
                    substitute_pair(B,pos,c.index,job.eventIndex, ...
                    jobs,events,concepts,failedJobs,nextUid,nextTime);
                concepts(c.index).useCount = concepts(c.index).useCount + 1;
                compressSuccesses = compressSuccesses + 1;
            end
        end
    end

    % Any successful compression can expose a new adjacency before the next
    % raw symbol arrives.
    [jobs,events,nextEventIndex] = scan_and_schedule( ...
        B,concepts,pairMap,jobs,events,nextEventIndex,nextTime, ...
        rawIndex,cfg);
end
end


% ======================================================================
% DISCOVERY / COMPRESSION SCHEDULING
% ======================================================================

function [jobs,events,nextEventIndex] = scan_and_schedule( ...
    B,concepts,pairMap,jobs,events,nextEventIndex,t,rawIndex,cfg)

% Canonicalize pending-job storage every time this routine is entered.
% This is cheap (the model is intentionally tiny) and eliminates MATLAB's
% 1x0-vs-0x1 deletion ambiguity completely.
jobs = jobs(:);

if ~isempty(jobs)
    for qjob = 1:numel(jobs)
        validate_job_scalar_fields(jobs(qjob),qjob);
    end
end

n = numel(B.token);
if n<2
    return;
end

% ----------------------- Known concepts -------------------------------
for i=1:n-1
    pair = [B.token(i),B.token(i+1)];
    key = pair_key(pair(1),pair(2));
    if isKey(pairMap,key)
        ci = pairMap(key);
        uids = [B.uid(i),B.uid(i+1)];
        if ~has_matching_job(jobs,1,ci,pair,uids)
            [jobs,events,nextEventIndex] = add_job( ...
                jobs,events,nextEventIndex,1,ci,pair,uids,t,rawIndex,cfg, ...
                concepts(ci).level,concepts(ci).rawSpan);
        end
    end
end

% ----------------------- Unknown concepts -----------------------------
% Candidate evidence lives only in B. There is no persistent count.
pairs = zeros(max(0,n-1),2);
for i=1:n-1
    pairs(i,:) = [B.token(i),B.token(i+1)];
end

for i=1:n-1
    pair = pairs(i,:);
    key = pair_key(pair(1),pair(2));
    if isKey(pairMap,key)
        continue;
    end

    % Optional causal ablation: concept tokens may still compress but may
    % not become constituents of new concepts.
    if ~cfg.recursiveReuse
        if token_level(pair(1),concepts)>0 || token_level(pair(2),concepts)>0
            continue;
        end
    end

    if has_pending_learning_for_pair(jobs,pair)
        continue;
    end

    second = 0;
    for j=i+2:n-1
        if pairs(j,1)==pair(1) && pairs(j,2)==pair(2)
            second = j;
            break;
        end
    end
    if second==0
        continue;
    end

    uids = [B.uid(i),B.uid(i+1),B.uid(second),B.uid(second+1)];
    predictedLevel = 1 + max(token_level(pair(1),concepts), ...
                             token_level(pair(2),concepts));
    predictedSpan = token_raw_span(pair(1),concepts) + ...
                    token_raw_span(pair(2),concepts);

    [jobs,events,nextEventIndex] = add_job( ...
        jobs,events,nextEventIndex,2,0,pair,uids,t,rawIndex,cfg, ...
        predictedLevel,predictedSpan);
end
end


function [jobs,events,nextEventIndex] = add_job( ...
    jobs,events,nextEventIndex,typeCode,conceptIndex,pair,uids, ...
    t,rawIndex,cfg,resultLevel,resultRawSpan)

e = empty_event();
e.eventIndex = nextEventIndex;
e.type = ternary(typeCode==1,'compress','learn');
e.requestTime = t;
e.requestTokenIndex = rawIndex;
e.completionTime = NaN;
e.outcome = 'pending';
e.failureReason = '';
e.conceptIndex = conceptIndex;
e.child1 = pair(1);
e.child2 = pair(2);
e.resultLevel = resultLevel;
e.resultRawSpan = resultRawSpan;
e.exactSourceLevel = 0;
events(end+1,1) = e; %#ok<AGROW>

j = empty_job();

% A job type is a categorical scalar by construction:
%   1 = compression of an already learned pair
%   2 = induction of a previously unknown pair.
% Keep these assertions here so any malformed job is caught at creation
% rather than much later inside a logical expression.
assert(isnumeric(typeCode) && isscalar(typeCode) && any(typeCode == [1 2]), ...
    'V8 internal error: typeCode must be scalar 1 or 2; got size %s.', ...
    mat2str(size(typeCode)));
assert(isnumeric(conceptIndex) && isscalar(conceptIndex), ...
    'V8 internal error: conceptIndex must be scalar; got size %s.', ...
    mat2str(size(conceptIndex)));

j.typeCode = double(typeCode);
j.conceptIndex = double(conceptIndex);
j.pair = reshape(pair,1,[]);
j.uids = reshape(uids,1,[]);
j.requestTime = t;
j.completionTime = t + cfg.processingTime;
j.eventIndex = nextEventIndex;

assert(isscalar(j.typeCode), ...
    'V8 internal error: a newly constructed job has nonscalar typeCode.');

% IMPORTANT MATLAB detail:
% jobs may become 1x0 after deleting its last element.  Using
% jobs(end+1,1)=j in that state writes at (2,1) and MATLAB silently creates
% a phantom (1,1) struct whose fields are empty.  Append by NUMEL instead.
jobs(numel(jobs)+1,1) = j; %#ok<AGROW>

nextEventIndex = nextEventIndex + 1;
end


% ======================================================================
% SUBSTITUTION / INVALIDATION
% ======================================================================

function [B,jobs,events,concepts,failedJobs,nextUid] = ...
    substitute_pair(B,pos,conceptIndex,currentEventIndex,jobs,events, ...
    concepts,failedJobs,nextUid,t)

uids = [B.uid(pos),B.uid(pos+1)];

% Competing jobs touching consumed active items fail.
for u = uids
    [jobs,events,concepts,failedJobs] = invalidate_jobs( ...
        jobs,events,concepts,u,t,'invalidated_by_compression',failedJobs, ...
        currentEventIndex);
end

c = concepts(conceptIndex);

B.token(pos) = c.token;
B.uid(pos) = nextUid;
B.rawSpan(pos) = B.rawSpan(pos)+B.rawSpan(pos+1);
B.level(pos) = c.level;
B.rawStart(pos) = B.rawStart(pos);
B.rawEnd(pos) = B.rawEnd(pos+1);
nextUid = nextUid + 1;

B.token(pos+1) = [];
B.uid(pos+1) = [];
B.rawSpan(pos+1) = [];
B.level(pos+1) = [];
B.rawStart(pos+1) = [];
B.rawEnd(pos+1) = [];
end


function [jobs,events,concepts,failedJobs] = invalidate_jobs( ...
    jobs,events,concepts,uid,t,reason,failedJobs,excludeEventIndex)

if nargin < 9
    excludeEventIndex = -1;
end

remove = false(numel(jobs),1);
for j=1:numel(jobs)
    if jobs(j).eventIndex==excludeEventIndex
        continue;
    end
    if any(jobs(j).uids==uid)
        ei = jobs(j).eventIndex;
        if strcmp(events(ei).outcome,'pending')
            events(ei).outcome = 'failure';
            events(ei).failureReason = reason;
            events(ei).completionTime = t;
            validate_job_scalar_fields(jobs(j),j);
            if isequal(jobs(j).typeCode,1) && jobs(j).conceptIndex>0 && ...
                    jobs(j).conceptIndex<=numel(concepts)
                concepts(jobs(j).conceptIndex).failureCount = ...
                    concepts(jobs(j).conceptIndex).failureCount + 1;
            end
            failedJobs = failedJobs + 1;
        end
        remove(j) = true;
    end
end
jobs(remove) = [];
jobs = jobs(:);  % canonical orientation; prevents 1x0 phantom append bug
end


% ======================================================================
% SUMMARY / VALIDATION
% ======================================================================

function s = summarize(concepts,H,E,initialConceptCount,learnSuccesses, ...
    compressSuccesses,failedJobs,cfg,source)

s = struct();
s.activeCapacity = cfg.activeCapacity;
s.tokenInterval = cfg.tokenInterval;
s.inputRate = 1/cfg.tokenInterval;
s.recurrenceLag = cfg.recurrenceLag;
s.sequenceLength = height(H);
s.initialConceptCount = initialConceptCount;
s.finalConceptCount = numel(concepts);
s.newConceptCount = numel(concepts)-initialConceptCount;
s.learnSuccesses = learnSuccesses;
s.compressSuccesses = compressSuccesses;
s.failedJobs = failedJobs;

if isempty(concepts)
    levels = 0;
    exactLevels = 0;
    rawSpans = 1;
else
    levels = [concepts.level];
    exactLevels = [concepts.exactSourceLevel];
    rawSpans = [concepts.rawSpan];
end

s.maximumLearnedLevel = max(levels);
s.maximumExactSourceLevel = max([0 exactLevels]);
s.maximumConceptRawSpan = max(rawSpans);
s.maximumContextAmplification = max(H.contextAmplification);
s.meanContextAmplification = mean(H.contextAmplification);
s.integratedLearnedDepth = mean(H.maxLearnedLevel);
s.integratedExactDepth = mean(H.maxExactSourceLevel);

if isfield(source,'depth')
    s.sourceDepth = source.depth;
    s.reachedFullExactDepth = s.maximumExactSourceLevel>=source.depth;
else
    s.sourceDepth = NaN;
    s.reachedFullExactDepth = false;
end

% First acquisition time/token by level.
maxReportLevel = max(8,max(s.maximumLearnedLevel, ...
    max(0,s.maximumExactSourceLevel)));
for L=1:maxReportLevel
    idx = find(H.maxLearnedLevel>=L,1,'first');
    if isempty(idx), v=NaN; else, v=H.tokenIndex(idx); end
    s.(sprintf('firstTokenLearnedLevel%d',L)) = v;

    idx = find(H.maxExactSourceLevel>=L,1,'first');
    if isempty(idx), v=NaN; else, v=H.tokenIndex(idx); end
    s.(sprintf('firstTokenExactLevel%d',L)) = v;
end

if ~isempty(E)
    learnMask = strcmp(E.type,'learn');
    succMask = strcmp(E.outcome,'success');
    s.successfulLearningEvents = sum(learnMask & succMask);
    if any(learnMask)
        s.learningEventSuccessFraction = ...
            sum(learnMask & succMask)/sum(learnMask);
    else
        s.learningEventSuccessFraction = NaN;
    end
else
    s.successfulLearningEvents = 0;
    s.learningEventSuccessFraction = NaN;
end

% A deliberately descriptive "cascade acceleration" diagnostic:
% compare new-concept formation rate in second vs first half. The paper
% should show the full curves rather than rely only on this scalar.
newC = H.newConceptCount;
n = height(H);
h = floor(n/2);
early = (newC(h)-newC(1))/max(1,h-1);
late = (newC(end)-newC(h))/max(1,n-h);
s.earlyConceptFormationRate = early;
s.lateConceptFormationRate = late;
if early>0
    s.cascadeAccelerationRatio = late/early;
else
    s.cascadeAccelerationRatio = NaN;
end
end


function validate_run(R,cfg)
H = R.history;
assert(all(H.bufferSlots<=cfg.activeCapacity));
assert(all(H.contextAmplification>=1-1e-12));

E = R.events;
if ~isempty(E)
    learnFail = strcmp(E.type,'learn') & strcmp(E.outcome,'failure');
    % Failed learning events must not appear as concepts. There is no
    % candidate-state field by design; the event log is diagnostic only.
    assert(all(~isnan(E.requestTime(learnFail))));
end

% Every confirmed concept is exactly a binary composition.
for k=1:numel(R.concepts)
    assert(R.concepts(k).rawSpan>=2);
end
end


% ======================================================================
% LOW-LEVEL HELPERS
% ======================================================================

function [ok,pos] = locate_pair(B,uid1,uid2,pair)
ok = false;
pos = 0;
i1 = find(B.uid==uid1,1);
i2 = find(B.uid==uid2,1);
if isempty(i1) || isempty(i2) || i2~=i1+1
    return;
end
if B.token(i1)==pair(1) && B.token(i2)==pair(2)
    ok = true;
    pos = i1;
end
end


function tf = has_matching_job(jobs,typeCode,conceptIndex,pair,uids)
tf = false;
for jj=1:numel(jobs)
    validate_job_scalar_fields(jobs(jj),jj);

    sameType = isequal(jobs(jj).typeCode,typeCode);
    sameConcept = isequal(jobs(jj).conceptIndex,conceptIndex);
    samePair = isequal(jobs(jj).pair,pair);
    sameUids = isequal(jobs(jj).uids,uids);

    if sameType && sameConcept && samePair && sameUids
        tf = true;
        return;
    end
end
end


function tf = has_pending_learning_for_pair(jobs,pair)
tf = false;
for jj=1:numel(jobs)
    validate_job_scalar_fields(jobs(jj),jj);

    isLearning = isequal(jobs(jj).typeCode,2);
    samePair = isequal(jobs(jj).pair,pair);

    if isLearning && samePair
        tf = true;
        return;
    end
end
end


function validate_job_scalar_fields(job,jobIndex)
% These fields are categorical/index scalars.  Pair and UID fields are
% intentionally vectors.
assert(isnumeric(job.typeCode) && isscalar(job.typeCode), ...
    ['V8 malformed job %d: typeCode must be a numeric scalar, ' ...
     'but its size is %s.'],jobIndex,mat2str(size(job.typeCode)));
assert(isnumeric(job.conceptIndex) && isscalar(job.conceptIndex), ...
    ['V8 malformed job %d: conceptIndex must be a numeric scalar, ' ...
     'but its size is %s.'],jobIndex,mat2str(size(job.conceptIndex)));
assert(isnumeric(job.eventIndex) && isscalar(job.eventIndex), ...
    ['V8 malformed job %d: eventIndex must be a numeric scalar, ' ...
     'but its size is %s.'],jobIndex,mat2str(size(job.eventIndex)));
end


function L = token_level(token,concepts)
L = 0;
if isempty(concepts)
    return;
end
idx = find([concepts.token]==token,1);
if ~isempty(idx)
    L = concepts(idx).level;
end
end


function n = token_raw_span(token,concepts)
n = 1;
if isempty(concepts)
    return;
end
idx = find([concepts.token]==token,1);
if ~isempty(idx)
    n = concepts(idx).rawSpan;
end
end


function x = combined_expansion(pair,concepts,source)
% Store explicit raw expansion only for the synthetic source. Arbitrary
% language concepts may grow very large, and exact source matching is not
% needed there.
if ~isfield(source,'exactExpansionKeys')
    x = [];
    return;
end
x1 = token_expansion(pair(1),concepts);
x2 = token_expansion(pair(2),concepts);
if isempty(x1) || isempty(x2)
    x = [];
else
    x = [x1 x2];
end
end


function x = token_expansion(token,concepts)
idx = [];
if ~isempty(concepts)
    idx = find([concepts.token]==token,1);
end
if isempty(idx)
    x = token;
else
    x = concepts(idx).rawExpansion;
end
end


function L = exact_source_level(expansion,source)
L = 0;
if isempty(expansion) || ~isfield(source,'exactExpansionKeys')
    return;
end
key = expansion_key(expansion);
for q=1:source.depth
    if any(strcmp(source.exactExpansionKeys{q},key))
        L = q;
        return;
    end
end
end


function B = remove_buffer_item(B,pos)
B.token(pos) = [];
B.uid(pos) = [];
B.rawSpan(pos) = [];
B.level(pos) = [];
B.rawStart(pos) = [];
B.rawEnd(pos) = [];
end


function B = empty_buffer()
B = struct();
B.token = zeros(0,1);
B.uid = zeros(0,1);
B.rawSpan = zeros(0,1);
B.level = zeros(0,1);
B.rawStart = zeros(0,1);
B.rawEnd = zeros(0,1);
end


function c = empty_concept()
c = struct('index',0,'token',0,'child1',0,'child2',0,'level',0, ...
    'rawSpan',0,'formationTime',NaN,'formationTokenIndex',NaN, ...
    'useCount',0,'failureCount',0,'inherited',false, ...
    'rawExpansion',[],'exactSourceLevel',0);
end


function j = empty_job()
j = struct('typeCode',0,'conceptIndex',0,'pair',[0 0], ...
    'uids',zeros(1,0),'requestTime',NaN,'completionTime',NaN, ...
    'eventIndex',0);
end


function e = empty_event()
e = struct('eventIndex',0,'type','','requestTime',NaN, ...
    'requestTokenIndex',NaN,'completionTime',NaN,'outcome','', ...
    'failureReason','','conceptIndex',0,'child1',0,'child2',0, ...
    'resultLevel',0,'resultRawSpan',0,'exactSourceLevel',0);
end


function T = events_to_table(events)
if isempty(events)
    T = table();
else
    T = struct2table(events);
end
end


function k = pair_key(a,b)
k = sprintf('%d_%d',a,b);
end


function k = expansion_key(x)
k = sprintf('%d,',x);
end


function y = ternary(cond,a,b)
if cond, y=a; else, y=b; end
end
