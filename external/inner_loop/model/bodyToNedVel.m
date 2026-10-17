classdef bodyToNedVel < matlab.System
    % bodyToNedVel
    % Convert body-frame velocity [u; v; w] to NED-frame velocity [vn; ve; vd].
    %
    % Coordinate convention:
    %   Body frame: x_b forward, y_b right, z_b down
    %   NED frame:  x_n North, y_n East, z_n Down
    %
    % Euler angle convention:
    %   phi   : roll angle,  rad
    %   theta : pitch angle, rad
    %   psi   : yaw angle,   rad
    %
    % Input:
    %   states = [V_body; Euler] = [u; v; w; phi; theta; psi]
    %
    % Output:
    %   V_ned = [vn; ve; vd]

    methods (Access = protected)
        function V_ned = stepImpl(~, states)
            %#codegen
            % states : [6 x 1] or [1 x 6]
            %          states = [u; v; w; phi; theta; psi]

            u = states(1);
            v = states(2);
            w = states(3);

            phi   = states(4);
            theta = states(5);
            psi   = states(6);

            cphi = cos(phi);
            sphi = sin(phi);

            ctheta = cos(theta);
            stheta = sin(theta);

            cpsi = cos(psi);
            spsi = sin(psi);

            % Direction cosine matrix from body frame to NED frame.
            C_nb = [ ...
                ctheta*cpsi,  sphi*stheta*cpsi - cphi*spsi,  cphi*stheta*cpsi + sphi*spsi;
                ctheta*spsi,  sphi*stheta*spsi + cphi*cpsi,  cphi*stheta*spsi - sphi*cpsi;
                -stheta,      sphi*ctheta,                    cphi*ctheta ];

            % Body frame to NED frame.
            V_ned = C_nb * [u; v; w];
        end

        function resetImpl(~)
            % No discrete states to reset.
        end

        function num = getNumInputsImpl(~)
            num = 1;
        end

        function num = getNumOutputsImpl(~)
            num = 1;
        end

        function varargout = getInputNamesImpl(~)
            varargout = {'states'};
        end

        function varargout = getOutputNamesImpl(~)
            varargout = {'V_ned'};
        end

        function flag = isInputSizeMutableImpl(~, ~)
            flag = false;
        end

        function flag = isInputDataTypeMutableImpl(~, ~)
            flag = false;
        end

        function sz = getOutputSizeImpl(~)
            sz = [3 1];
        end

        function dt = getOutputDataTypeImpl(~)
            dt = 'double';
        end

        function cp = isOutputComplexImpl(~)
            cp = false;
        end

        function fs = isOutputFixedSizeImpl(~)
            fs = true;
        end
    end
end