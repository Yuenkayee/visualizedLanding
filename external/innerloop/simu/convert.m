classdef convert < matlab.System
    % Fixed camera centre and neutral geometry; measured rotation varies.
    % Fill all three Nontunable properties from calibrated installation data.
    properties (Nontunable)
        T_body_camera = nan(4,4)
        H_ship_body = nan(3,1)
        gear_body = nan(3,1)
    end
    methods (Access=protected)
        function setupImpl(obj,varargin)
            validateattributes(obj.T_body_camera,{'double'},{'size',[4 4],'finite','real'},'', 'T_body_camera');
            validateattributes(obj.H_ship_body,{'double'},{'size',[3 1],'finite','real'},'', 'H_ship_body');
            validateattributes(obj.gear_body,{'double'},{'size',[3 1],'finite','real'},'', 'gear_body');
            R=obj.T_body_camera(1:3,1:3);
            assert(norm(R'*R-eye(3),'fro')<1e-6 && det(R)>0, ...
                'landing:Geometry','T_body_camera rotation must be SO(3)');
            assert(norm(obj.T_body_camera(4,:)-[0 0 0 1])<1e-8, ...
                'landing:Geometry','Invalid homogeneous transform');
        end
        function varargout=stepImpl(obj,t,heli,ship,nav_T,nav_v,nav_P,nav_ok,nav_t,q,Rbc,wg,Pg,okg,tg,wnav)
            [varargout{1:15}]=innerLoop_conversion(t,heli,ship, ...
                obj.T_body_camera,obj.H_ship_body,obj.gear_body, ...
                nav_T,nav_v,nav_P,nav_ok,nav_t,q,Rbc,wg,Pg,okg,tg,wnav);
        end
        function n=getNumInputsImpl(~), n=15; end
        function n=getNumOutputsImpl(~), n=15; end
        function varargout=getOutputSizeImpl(~)
            varargout={[4 4],[4 4],[3 1],[3 1],[3 1],[3 1], ...
                [3 1],[1 1],[3 3],[1 1],[3 1],[1 1],[4 4],[4 4],[3 1]};
        end
        function varargout=getOutputDataTypeImpl(~)
            varargout=repmat({'double'},1,15);
        end
        function varargout=isOutputComplexImpl(~)
            varargout=repmat({false},1,15);
        end
        function varargout=isOutputFixedSizeImpl(~)
            varargout=repmat({true},1,15);
        end
        function varargout=getInputNamesImpl(~)
            varargout={'timestamp','helicopter_state','ship_state', ...
                'nav_T_deck_camera','nav_velocity_deck','nav_covariance', ...
                'nav_healthy','nav_timestamp','gimbal_angles_rad','nav_R_body_camera', ...
                'nav_gimbal_omega_camera','nav_gimbal_covariance','nav_gimbal_valid', ...
                'nav_gimbal_timestamp','nav_angular_velocity_camera'};
        end
        function varargout=getOutputNamesImpl(~)
            varargout={'T_world_deck','T_deck_camera','velocity_deck', ...
                'helicopter_velocity_ned_mps','ship_velocity_ned_mps', ...
                'gear_relative_truth_ned','gear_relative_estimate_ned', ...
                'relative_distance_estimate_m','relative_covariance_ned', ...
                'feedback_valid','navigation_velocity_camera_ned_mps','state_timestamp', ...
                'T_body_camera_actual','nav_T_deck_body','navigation_velocity_gear_ned_mps'};
        end
    end
end
