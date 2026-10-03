classdef uvw_Gen < matlab.System
    % uvw_Gen
    % Generate the commanded relative velocity of the helicopter with
    % respect to the ship in the NED frame from relative position error.
    %
    % Input:
    %   Delta_r = r_ship - r_heli = [Delta_x; Delta_y; Delta_z]
    %
    % Output:
    %   Delta_v = [Delta_v_n; Delta_v_e; Delta_v_d]
    %
    % Control law:
    %   Delta_v_i = maxDeltaVel_i * atan(Delta_i), i = n,e,d
    %
    % Notes:
    %   Delta_x, Delta_y, Delta_z are assumed to correspond to the NED
    %   relative position components. The output is therefore the commanded
    %   NED relative velocity.

    properties
        % Maximum commanded relative velocity in NED north direction.

        maxDeltaVel_n = 1.0

        % Maximum commanded relative velocity in NED east direction.

        maxDeltaVel_e = 1.0

        % Maximum commanded relative velocity in NED down direction.
        
        maxDeltaVel_d = 1.0
    end

    methods (Access = protected)
        function Delta_v = stepImpl(obj, Delta_r)
            % Ensure the input is treated as a column vector.
            Delta_r = Delta_r(:);

            % Input protection for Simulink/MATLAB System usage.
            if numel(Delta_r) ~= 3
                error('uvw_Gen:InvalidInputSize', ...
                    'Input Delta_r must be a 3-element vector [Delta_x; Delta_y; Delta_z].');
            end

            Delta_x = Delta_r(1);
            Delta_y = Delta_r(2);
            Delta_z = Delta_r(3);

            Delta_v_n = obj.maxDeltaVel_n * atan(Delta_x);
            Delta_v_e = obj.maxDeltaVel_e * atan(Delta_y);
            Delta_v_d = obj.maxDeltaVel_d * atan(Delta_z);

            Delta_v = [Delta_v_n; Delta_v_e; Delta_v_d];
        end

        function resetImpl(~)
            % No discrete states to reset.
        end

        function out = getOutputSizeImpl(~)
            out = [3 1];
        end

        function out = getOutputDataTypeImpl(~)
            out = 'double';
        end

        function out = isOutputComplexImpl(~)
            out = false;
        end

        function out = isOutputFixedSizeImpl(~)
            out = true;
        end

        function sz = getInputSizeImpl(~)
            sz = [3 1];
        end

        function dt = getInputDataTypeImpl(~)
            dt = 'double';
        end

        function c = isInputComplexImpl(~)
            c = false;
        end

        function f = isInputFixedSizeImpl(~)
            f = true;
        end
    end
end