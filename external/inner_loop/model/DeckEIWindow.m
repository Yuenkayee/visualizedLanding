classdef DeckEIWindow < matlab.System
    % Inclusive 50 Hz forecast interval; runtime configuration also sets
    % the predictor wave clock's lead to the upper endpoint.
    properties (Nontunable)
        WindowSamples (1,1) double {mustBeInteger,mustBePositive} = 126
    end
    properties (DiscreteState)
        samples
        cursor
    end
    methods (Access=protected)
        function setupImpl(obj,~)
            resetImpl(obj);
        end
        function resetImpl(obj)
            obj.samples = repmat(realmax,obj.WindowSamples,1);
            obj.cursor = 1;
        end
        function peak = stepImpl(obj,ei)
            if ~isfinite(ei) || ei < 0
                ei = realmax;
            end
            obj.samples(obj.cursor) = ei;
            obj.cursor = mod(obj.cursor,obj.WindowSamples)+1;
            peak = max(obj.samples);
        end
        function st = getSampleTimeImpl(obj)
            st = createSampleTime(obj,'Type','Discrete','SampleTime',0.02);
        end
    end
end
