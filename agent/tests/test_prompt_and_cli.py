from __future__ import annotations

import os
import unittest
from unittest.mock import patch

from agent.cli import build_parser
from agent.config import load
from agent.robotwin.prompts import generic_action_space, observation_prompt, system_prompt, task_prompt


class PromptAndCliTest(unittest.TestCase):
    def test_provider_defaults_use_large_images_256k_and_default_temperature(self) -> None:
        args = build_parser().parse_args(["config"])

        with patch.dict(os.environ, {}, clear=True):
            config = load(args)

        self.assertEqual(config.model, "openai/gpt-5.6-sol")
        self.assertEqual(config.wire_api, "chat_completions")
        self.assertIsNone(config.temperature)
        self.assertEqual(config.reasoning_effort, "medium")
        self.assertEqual(config.max_tokens, 4096)
        self.assertTrue(config.stream)
        self.assertEqual(config.anthropic_thinking, "enabled")
        self.assertEqual(config.anthropic_thinking_budget_tokens, 1024)
        self.assertEqual(config.request_image_max_width, 1280)
        self.assertEqual(config.request_image_max_height, 960)
        self.assertEqual(config.request_image_jpeg_quality, 95)
        self.assertEqual(config.request_max_images, 4)
        self.assertEqual(config.transient_http_retry_attempts, 6)
        self.assertEqual(config.transient_http_retry_backoff_seconds, 1.5)
        self.assertIsNone(config.quota_fallback_base_url)
        self.assertIsNone(config.quota_fallback_model)
        self.assertTrue(config.context_compaction_enabled)
        self.assertEqual(config.context_window_tokens, 262_144)
        self.assertEqual(config.context_compaction_threshold, 0.80)
        self.assertEqual(config.context_compaction_keep_recent_turns, 6)
        self.assertEqual(config.context_compaction_keep_recent_images, 4)

    def test_cli_accepts_max_tokens(self) -> None:
        args = build_parser().parse_args(["--max-tokens", "1024", "config"])

        with patch.dict(os.environ, {}, clear=True):
            config = load(args)

        self.assertEqual(config.max_tokens, 1024)

    def test_robotwin_step_default_and_explicit_override(self) -> None:
        parser = build_parser()
        for command in ("eval", "loop"):
            self.assertEqual(parser.parse_args([command, "robotwin"]).max_steps, 100)
            self.assertEqual(
                parser.parse_args([command, "robotwin", "--max-steps", "50"]).max_steps,
                50,
            )

    def test_explicit_small_image_settings_override_new_defaults(self) -> None:
        args = build_parser().parse_args(["config"])
        with patch.dict(os.environ, {
            "AGENT_REQUEST_IMAGE_MAX_WIDTH": "384",
            "AGENT_REQUEST_IMAGE_MAX_HEIGHT": "288",
            "AGENT_REQUEST_IMAGE_JPEG_QUALITY": "60",
            "AGENT_CONTEXT_WINDOW_TOKENS": "65536",
            "AGENT_MAX_TOKENS": "8192",
            "AGENT_TEMPERATURE": "0",
        }, clear=True):
            config = load(args)
        self.assertEqual((config.request_image_max_width, config.request_image_max_height), (384, 288))
        self.assertEqual(config.request_image_jpeg_quality, 60)
        self.assertEqual(config.context_window_tokens, 65536)
        self.assertEqual(config.max_tokens, 8192)
        self.assertEqual(config.temperature, 0.0)

    def test_environment_can_omit_temperature_from_provider_payload(self) -> None:
        from agent.client import ChatClient
        from agent.session import Message, Part

        args = build_parser().parse_args(["--wire-api", "anthropic_messages", "config"])
        with patch.dict(os.environ, {"AGENT_TEMPERATURE": "none"}, clear=True):
            config = load(args)

        self.assertIsNone(config.temperature)
        payload = ChatClient(config).payload(
            [Message(role="user", content=[Part.text_part("OK")])]
        )
        self.assertNotIn("temperature", payload)

    def test_explicit_temperature_zero_overrides_omitted_environment_temperature(self) -> None:
        args = build_parser().parse_args(["--temperature", "0", "config"])
        with patch.dict(os.environ, {"AGENT_TEMPERATURE": "none"}, clear=True):
            config = load(args)
        self.assertEqual(config.temperature, 0.0)

    def test_environment_overrides_default_max_tokens(self) -> None:
        args = build_parser().parse_args(["config"])

        with patch.dict(
            os.environ,
            {"AGENT_MAX_TOKENS": "4096"},
            clear=True,
        ):
            config = load(args)

        self.assertEqual(config.max_tokens, 4096)

    def test_environment_configures_quota_balance_fallback(self) -> None:
        args = build_parser().parse_args(["config"])

        with patch.dict(
            os.environ,
            {
                "AGENT_QUOTA_FALLBACK_BASE_URL": "https://balance.example/v3/",
                "AGENT_QUOTA_FALLBACK_MODEL": "balance-model",
                "AGENT_QUOTA_FALLBACK_API_KEY": "sk-balance",
                "AGENT_QUOTA_FALLBACK_COOLDOWN_SECONDS": "120",
            },
            clear=True,
        ):
            config = load(args)

        self.assertEqual(config.quota_fallback_base_url, "https://balance.example/v3")
        self.assertEqual(config.quota_fallback_model, "balance-model")
        self.assertEqual(config.quota_fallback_api_key, "sk-balance")
        self.assertEqual(config.quota_fallback_cooldown_seconds, 120.0)
        self.assertEqual(config.redacted["quota_fallback_api_key"], "***")

    def test_quota_fallback_requires_url_and_model_together(self) -> None:
        args = build_parser().parse_args(["config"])

        with patch.dict(
            os.environ,
            {"AGENT_QUOTA_FALLBACK_BASE_URL": "https://balance.example/v3"},
            clear=True,
        ):
            with self.assertRaisesRegex(ValueError, "configured together"):
                load(args)

    def test_cli_accepts_responses_wire_api_and_disables_reasoning_parameter(self) -> None:
        args = build_parser().parse_args(
            ["--wire-api", "responses", "--reasoning-effort", "none", "config"]
        )

        with patch.dict(os.environ, {}, clear=True):
            config = load(args)

        self.assertEqual(config.wire_api, "responses")
        self.assertIsNone(config.reasoning_effort)

    def test_cli_accepts_anthropic_messages_wire_api(self) -> None:
        args = build_parser().parse_args(
            [
                "--wire-api",
                "anthropic_messages",
                "--anthropic-version",
                "2023-06-01",
                "--reasoning-effort",
                "none",
                "config",
            ]
        )

        with patch.dict(os.environ, {}, clear=True):
            config = load(args)

        self.assertEqual(config.wire_api, "anthropic_messages")
        self.assertEqual(config.anthropic_version, "2023-06-01")
        self.assertIsNone(config.reasoning_effort)

    def test_anthropic_environment_aliases_select_messages_and_auth_token(self) -> None:
        args = build_parser().parse_args(["config"])

        with patch.dict(
            os.environ,
            {
                "ANTHROPIC_BASE_URL": "https://gateway.example",
                "ANTHROPIC_MODEL": "claude-opus-test",
                "ANTHROPIC_AUTH_TOKEN": "auth-token-test",
            },
            clear=True,
        ):
            config = load(args)

        self.assertEqual(config.base_url, "https://gateway.example/v1")
        self.assertEqual(config.model, "claude-opus-test")
        self.assertEqual(config.wire_api, "anthropic_messages")
        self.assertEqual(config.auth_token, "auth-token-test")
        self.assertIsNone(config.api_key)
        self.assertEqual(config.redacted["auth_token"], "***")

    def test_cli_accepts_context_compaction_controls(self) -> None:
        args = build_parser().parse_args(
            [
                "--context-window-tokens",
                "100000",
                "--context-compaction-threshold",
                "0.75",
                "--context-compaction-keep-recent-turns",
                "4",
                "--context-compaction-keep-recent-images",
                "2",
                "config",
            ]
        )

        with patch.dict(os.environ, {}, clear=True):
            config = load(args)

        self.assertEqual(config.context_window_tokens, 100_000)
        self.assertEqual(config.context_compaction_threshold, 0.75)
        self.assertEqual(config.context_compaction_keep_recent_turns, 4)
        self.assertEqual(config.context_compaction_keep_recent_images, 2)

    def test_cli_disable_context_compaction_overrides_environment(self) -> None:
        args = build_parser().parse_args(["--no-context-compaction", "config"])

        with patch.dict(
            os.environ,
            {"AGENT_CONTEXT_COMPACTION_ENABLED": "true"},
            clear=True,
        ):
            config = load(args)

        self.assertFalse(config.context_compaction_enabled)

    def test_cli_defaults_to_generic_local_constraint_profile(self) -> None:
        args = build_parser().parse_args(["eval", "robotwin"])

        self.assertEqual(args.local_constraint_profile, "generic")

    def test_cli_defaults_to_clean_model_view_and_debug_overlays(self) -> None:
        args = build_parser().parse_args(["eval", "robotwin"])

        self.assertEqual(args.model_overlay, "none")
        self.assertEqual(args.model_debug_overlay, "eepose")
        self.assertEqual(args.record_overlay, "eepose")

    def test_cli_accepts_intermediate_camera_views(self) -> None:
        args = build_parser().parse_args(
            [
                "eval",
                "robotwin",
                "--model-view",
                "front_side_45",
                "--record-view",
                "side_top_45",
                "--initial-camera-view",
                "oblique_45",
            ]
        )

        self.assertEqual(args.model_view, "front_side_45")
        self.assertEqual(args.record_view, "side_top_45")
        self.assertEqual(args.initial_camera_view, "oblique_45")

    def test_cli_accepts_dual_arm_mode(self) -> None:
        args = build_parser().parse_args(["eval", "robotwin", "--task", "lift_pot", "--active-arm", "both"])
        self.assertEqual(args.active_arm, "both")

    def test_cli_and_system_prompt_accept_response_language(self) -> None:
        args = build_parser().parse_args(["--response-language", "English", "eval", "robotwin"])
        self.assertEqual(args.response_language, "English")
        prompt = system_prompt(
            "- numeric controls",
            task="grasp_single_bottle_upright",
            model_view="active",
            record_view="side",
            camera_policy="fine",
            response_language="English",
        )
        self.assertIn("Use English only", prompt)
        self.assertIn("context-checkpoint summaries", prompt)

    def test_system_prompt_includes_dual_arm_protocol_only_in_dual_mode(self) -> None:
        prompt = system_prompt(
            "Dual-arm mode:\n- synchronized controls",
            task="lift_pot",
            model_view="active",
            model_overlay="none",
            record_view="side",
            record_overlay="eepose",
            camera_policy="fine",
        )
        self.assertIn('"arm":"left|right"', prompt)
        self.assertIn('"action":"dual_gripper.move_world"', prompt)
        self.assertNotIn(
            '{"action":"gripper.move_world","axis":"x|y|z"',
            prompt,
        )

    def test_system_prompt_injects_horizontal_block_handover_rules(self) -> None:
        prompt = system_prompt(
            "Dual-arm mode:\n- synchronized controls",
            task="handover_horizontal_block",
            model_view="active",
            model_overlay="none",
            record_view="gripper_follow",
            record_overlay="eepose",
            camera_policy="fine",
        )

        self.assertIn("[dynamic_giver_selection]", prompt)
        self.assertIn("must not become a fixed role across seeds", prompt)
        self.assertIn("[opposite_receiver_assignment]", prompt)
        self.assertIn("[verticalize_before_handover]", prompt)
        self.assertIn("overrides any expert demonstration", prompt)
        self.assertIn("[opposed_receiver_approach]", prompt)
        self.assertIn("must not approach the transfer from the same side", prompt)
        self.assertIn("[receiver_side_grasp_posture]", prompt)
        self.assertIn("approach along the receiver's actual local approach direction", prompt)
        self.assertIn("[handover_contact_recovery]", prompt)
        self.assertIn("displaces the vertical block", prompt)
        self.assertIn("[close_before_release]", prompt)

    def test_system_prompt_injects_lift_pot_mirrored_handle_rules(self) -> None:
        prompt = system_prompt(
            "Dual-arm mode:\n- synchronized controls",
            task="lift_pot",
            model_view="active",
            model_overlay="none",
            record_view="gripper_follow",
            record_overlay="eepose",
            camera_policy="fine",
        )

        self.assertIn("Task profile: mirrored dual-handle pot lift", prompt)
        self.assertIn("[one_side_handle_per_arm]", prompt)
        self.assertIn("[mirrored_outside_in_grasp_posture]", prompt)
        self.assertIn("must not retain a shared forward-facing", prompt)
        self.assertIn("[rotate_high_before_handle_approach]", prompt)
        self.assertIn("[outside_height_then_radial_insertion]", prompt)
        self.assertIn("[verify_both_handle_wraps]", prompt)
        self.assertIn("[synchronized_close_and_level_lift]", prompt)

    def test_system_prompt_injects_generalization_policy_only_for_held_out_tasks(self) -> None:
        generalization_prompt = system_prompt(
            "- numeric controls",
            task="grasp_single_bottle_generalization",
            model_view="active",
            model_overlay="none",
            record_view="side",
            record_overlay="eepose",
            camera_policy="fine",
        )
        regular_prompt = system_prompt(
            "- numeric controls",
            task="grasp_single_bottle",
            model_view="active",
            model_overlay="none",
            record_view="side",
            record_overlay="eepose",
            camera_policy="fine",
        )

        self.assertIn("Generalization evaluation policy", generalization_prompt)
        self.assertIn("[mandatory_protocol_and_objective]", generalization_prompt)
        self.assertIn("[expert_is_transferable_reference]", generalization_prompt)
        self.assertIn("not an exact trajectory to replay", generalization_prompt)
        self.assertIn("[no_expert_appearance_assumptions]", generalization_prompt)
        self.assertIn("colors, labels, brands", generalization_prompt)
        self.assertIn("[preserve_transferable_safety]", generalization_prompt)
        self.assertNotIn("Generalization evaluation policy", regular_prompt)

    def test_cli_rejects_task_specific_local_constraint_profile(self) -> None:
        with self.assertRaises(SystemExit):
            build_parser().parse_args(["eval", "robotwin", "--local-constraint-profile", "horizontal_bottle_topdown"])

    def test_cli_accepts_prelearned_expert_constraints(self) -> None:
        args = build_parser().parse_args(["eval", "robotwin", "--expert-learning-file", "learning.json"])

        self.assertEqual(args.expert_learning_file, "learning.json")

    def test_cli_accepts_summary_experiment_mode(self) -> None:
        args = build_parser().parse_args(
            ["eval", "robotwin", "--experiment-mode", "shared-summary"]
        )

        self.assertEqual(args.experiment_mode, "shared-summary")

    def test_cli_accepts_learned_constraint_file_for_eval(self) -> None:
        args = build_parser().parse_args(["eval", "robotwin", "--learned-constraint-file", "constraints.json"])

        self.assertEqual(args.learned_constraint_file, "constraints.json")

    def test_cli_accepts_robotwin_loop_auto_correction(self) -> None:
        args = build_parser().parse_args(["loop", "robotwin", "--max-rounds", "3", "--correction", "auto"])

        self.assertEqual(args.command, "loop")
        self.assertEqual(args.loop_command, "robotwin")
        self.assertEqual(args.max_rounds, 3)
        self.assertEqual(args.correction, "auto")

    def test_generic_action_space_filters_task_specific_bottle_rules(self) -> None:
        raw = "\n".join(
            [
                "- Gripper translation actions use fixed world XYZ axes.",
                "- Gripper translation scales: world_x_*/world_y_* support small=0.015 m, medium=0.040 m, large=0.100 m.",
                "- Gripper rotation sign convention: rotate_{axis}_ccw is positive; small=8 deg, large=45 deg.",
                "- For bottle grasping, do not lower or close based on center overlap alone.",
                "- Treat the latest gripper.rotate_* as a rotate-lock for the final approach.",
                "- Camera actions affect only the observation viewpoint.",
            ]
        )

        filtered = generic_action_space(raw)

        self.assertIn("fixed world XYZ", filtered)
        self.assertIn("Camera actions", filtered)
        self.assertNotIn("For bottle grasping", filtered)
        self.assertNotIn("rotate-lock", filtered)
        self.assertNotIn("Gripper translation scales", filtered)
        self.assertNotIn("small=8 deg", filtered)

    def test_system_prompt_keeps_protocol_and_removes_hard_bottle_gate(self) -> None:
        prompt = system_prompt(
            "- For bottle grasping, do not lower or close based on center overlap alone.\n"
            "- Gripper rotation actions rotate around fixed world axes.",
            task="grasp_single_bottle",
            model_view="active",
            model_overlay="none",
            record_view="side",
            record_overlay="eepose",
            camera_policy="fine",
        )

        self.assertIn('"action":"gripper.move_world"', prompt)
        self.assertIn('"action":"gripper.rotate_local"', prompt)
        self.assertIn('"tool":"expert.learn"', prompt)
        self.assertIn('"affordance_profile"', prompt)
        self.assertIn('"part_shape"', prompt)
        self.assertIn('{"tool":"expert.retrieve","args":{"mode":"trajectory"}', prompt)
        self.assertNotIn('"seed":0', prompt)
        self.assertNotIn('"phase":"final_grasp"', prompt)
        self.assertNotIn('"action":"gripper.rotate_world"', prompt)
        self.assertNotIn("Infer the grasp strategy", prompt)
        self.assertNotIn("fixed world axes", prompt)
        self.assertIn("distance_mm is 1..100", prompt)
        self.assertIn("current local axis", prompt)
        self.assertIn("Constraint lifecycle", prompt)
        self.assertIn("revise or remove constraints that evidence shows are wrong or harmful", prompt)
        self.assertIn("[constraint_lifecycle]", prompt)
        self.assertIn("[expert_grasp_position]", prompt)
        self.assertIn("no key frame", prompt)
        self.assertIn("required frame count", prompt)
        self.assertIn("call expert.learn", prompt)
        self.assertIn("demonstrated object part", prompt)
        self.assertIn("[pre_action_grasp_point_analysis]", prompt)
        self.assertIn("This is model reasoning, not a separate tool or local gate", prompt)
        self.assertIn("[stable_grasp_region_centering]", prompt)
        self.assertIn("Grasp-region selection and insertion depth are separate requirements", prompt)
        self.assertIn("center of a broad, regular, unobstructed graspable region", prompt)
        self.assertIn("along the object's visible length and surface directions", prompt)
        self.assertIn("deep insertion of the wrong edge region", prompt)
        self.assertNotIn('"tool":"grasp.plan"', prompt)
        self.assertIn("Preserve all existing validated constraints", prompt)
        self.assertIn("[full_wrap_before_close]", prompt)
        self.assertIn("deeply enclosed in the effective gripping region", prompt)
        self.assertIn("keep the gripper open and continue advancing", prompt)
        self.assertIn("[grasp_center_depth_before_close]", prompt)
        self.assertIn("both lateral straddling and insertion depth", prompt)
        self.assertIn("as close as safely possible to the gripper center", prompt)
        self.assertIn("[continuous_insertion_toward_gc]", prompt)
        self.assertIn("only after entering the final insertion phase", prompt)
        self.assertIn("target is already aligned inside the open gap", prompt)
        self.assertIn("Do not apply it during initial approach", prompt)
        self.assertIn("repeatedly try to insert the open gripper", prompt)
        self.assertIn("top-down grasp", prompt)
        self.assertIn("horizontal or side grasp", prompt)
        self.assertIn("target is already visibly close to GC", prompt)
        self.assertIn("cannot advance farther", prompt)
        self.assertIn("Fingertip contact is a cue to continue inserting", prompt)
        self.assertIn("target center must be near GC", prompt)
        self.assertNotIn("[negative_evidence_veto_before_close]", prompt)
        self.assertIn("[low_object_support_depth]", prompt)
        self.assertIn("the fingertip bottoms have reached or are touching the support plane", prompt)
        self.assertIn("the target has reached or contacted GC", prompt)
        self.assertIn("no further descent was achieved or is feasible", prompt)
        self.assertNotIn("[upright_object_wrap_confirmation]", prompt)
        self.assertIn("[complete_lift_after_close]", prompt)
        self.assertIn("Do not stop after a preliminary 20-30 mm rise", prompt)
        self.assertIn("environment reports success", prompt)
        self.assertIn("field name test_lift_rule is retained for protocol compatibility", prompt)
        self.assertNotIn("prefer a short test lift", prompt)
        self.assertIn("[clean_model_image]", prompt)
        self.assertIn("[active_perception]", prompt)
        self.assertIn("Use camera.view_topdown for world-XY alignment", prompt)
        self.assertIn("camera.view_front_side_45 to break horizontal front/side occlusion", prompt)
        self.assertIn("camera.view_side_top_45 to jointly inspect lateral alignment", prompt)
        self.assertIn("camera.view_oblique_45", prompt)
        self.assertIn("Do not restrict view selection to topdown and side", prompt)
        self.assertIn("oblique overlap alone is not final grasp proof", prompt)
        self.assertIn("[camera_action_budget]", prompt)
        self.assertIn("more than 5 consecutive camera actions", prompt)
        self.assertIn("A non-action tool call does not reset this count", prompt)
        self.assertIn("never change world-axis gripper controls", prompt)
        self.assertIn("rather than following a fixed view order", prompt)
        self.assertIn("treat evidence about changed dimensions as stale", prompt)
        self.assertIn("model overlay: none", prompt)
        self.assertIn("no EE/GC labels", prompt)
        self.assertIn("Record/debug overlays are human artifacts", prompt)
        self.assertNotIn("do not lower or close based on center overlap", prompt)
        self.assertNotIn("Task profile: horizontal bottle body grasp", prompt)
        self.assertNotIn("[descend_until_side_straddle]", prompt)
        self.assertNotIn("retreat slightly upward, adjust lateral/XY placement", prompt)

    def test_horizontal_bottle_prompt_includes_diagnosed_failure_constraints(self) -> None:
        prompt = task_prompt("grasp_single_bottle")

        self.assertIn("Do not treat topdown 2D overlap", prompt)
        self.assertIn("Constraint lifecycle", prompt)
        self.assertIn("[top_down_default_strategy]", prompt)
        self.assertIn("[rotate_lock_before_descent]", prompt)
        self.assertIn("[descend_until_side_straddle]", prompt)
        self.assertIn("[expert_grasp_region_check]", prompt)
        self.assertIn("top-down body grasp", prompt)
        self.assertIn("rotate_local", prompt)
        self.assertIn("local ry", prompt)
        self.assertIn("topdown_posture_ok_30deg is false, do not descend or close", prompt)
        self.assertIn("physical fingertips/finger pads", prompt)
        self.assertIn("two finger pads on opposite lateral sides", prompt)
        self.assertIn("top contact, top-side contact", prompt)
        self.assertIn("retreat slightly upward, adjust lateral/XY placement", prompt)
        self.assertIn("do not repeat a depth-only correction", prompt)
        self.assertIn("rounded/bumpy ends", prompt)
        self.assertIn("visible corrective change", prompt)
        self.assertIn("not imagined guide lines", prompt)
        self.assertIn("demonstrated grasp region", prompt)
        self.assertIn("do not replace the rotation, multi-view, side-depth, or completion-lift constraints", prompt)
        self.assertIn("world_z_neg", prompt)
        self.assertIn("10-40 mm", prompt)
        self.assertIn("lift directly along world +Z", prompt)
        self.assertIn("environment reports success", prompt)
        self.assertNotIn("20-30 mm world_z_pos test lift", prompt)

    def test_observation_prompt_does_not_auto_inject_task_profile(self) -> None:
        prompt = observation_prompt(
            {"task": "grasp_single_bottle", "gripper_state": {}, "action_history": []},
            step=0,
            model_view="active",
            model_overlay="none",
            record_view="side",
            record_overlay="eepose",
            initial_camera_view="center_high",
            camera_policy="fine",
            geometry={},
            available_actions=["camera.view_side", "gripper.close"],
            historical_steps=[],
        )

        self.assertIn("Task instruction: grasp_single_bottle", prompt)
        self.assertIn("Discrete available actions", prompt)
        self.assertNotIn("Task profile: horizontal bottle body grasp", prompt)
        self.assertNotIn("[descend_until_side_straddle]", prompt)

    def test_upright_bottle_prompt_does_not_use_horizontal_specific_rules(self) -> None:
        prompt = task_prompt("grasp_single_bottle_upright")

        self.assertIn("upright bottle", prompt)
        self.assertIn("[expert_grasp_region_check]", prompt)
        self.assertIn("demonstrated grasp height", prompt)
        self.assertNotIn("horizontal bottle", prompt)
        self.assertNotIn("topdown 2D overlap", prompt)

    def test_handover_mic_prompt_uses_microphone_and_vertical_transfer(self) -> None:
        prompt = task_prompt("handover_mic")

        self.assertIn("dynamically assigned microphone handover", prompt)
        self.assertIn("[dynamic_giver_selection]", prompt)
        self.assertIn("microphone's initial position", prompt)
        self.assertIn("must not become a fixed role across seeds", prompt)
        self.assertIn("[opposite_receiver_assignment]", prompt)
        self.assertIn("opposite arm as the receiver", prompt)
        self.assertIn("[verticalize_before_handover]", prompt)
        self.assertIn("microphone remains horizontal or diagonal", prompt)
        self.assertIn("[close_before_release]", prompt)
        self.assertIn("before opening or retreating the giver", prompt)

    def test_horizontal_block_prompt_is_separate_and_requires_vertical_transfer(self) -> None:
        prompt = task_prompt("handover_horizontal_block")

        self.assertIn("dynamically assigned horizontal-block handover", prompt)
        self.assertIn("[dynamic_giver_selection]", prompt)
        self.assertIn("horizontal block's initial position", prompt)
        self.assertIn("[opposite_receiver_assignment]", prompt)
        self.assertIn("[verticalize_before_handover]", prompt)
        self.assertIn("long axis is approximately world vertical", prompt)
        self.assertIn("receiver must not approach", prompt.lower())
        self.assertIn("[close_before_release]", prompt)
        self.assertNotIn("microphone's initial position", prompt)

    def test_handover_block_prompt_requires_transfer_then_right_arm_place(self) -> None:
        prompt = task_prompt("handover_block")

        self.assertIn("left-to-right block handover", prompt)
        self.assertIn("[fixed_handover_roles]", prompt)
        self.assertIn("left arm as the giver", prompt)
        self.assertIn("[receiver_before_giver_release]", prompt)
        self.assertIn("before opening or retreating the left gripper", prompt)
        self.assertIn("[right_arm_places_on_blue_pad]", prompt)
        self.assertIn("both grippers open", prompt)

    def test_cube_target_handover_prompt_assigns_roles_from_layout(self) -> None:
        prompt = task_prompt("handover_cube_to_target")

        self.assertIn("dynamically assigned two-stage cube relay", prompt)
        self.assertIn("[infer_roles_from_both_locations]", prompt)
        self.assertIn("arm on the cube's side as the giver", prompt)
        self.assertIn("arm on the target's side as the receiver", prompt)
        self.assertIn("never copy a fixed left/right assignment", prompt)
        self.assertIn("[mandatory_middle_handover]", prompt)
        self.assertIn("must not carry the cube directly", prompt)
        self.assertIn("[release_before_receiver_pickup]", prompt)
        self.assertIn("giver has opened and cleared", prompt)
        self.assertIn("resting cube as a new grasp", prompt)
        self.assertIn("[receiver_places_on_target]", prompt)

    def test_lift_pot_prompt_is_object_relative_and_has_no_hidden_pose(self) -> None:
        prompt = task_prompt("lift_pot")

        self.assertIn("mirrored dual-handle pot lift", prompt)
        self.assertIn("outside toward the pot center", prompt)
        self.assertIn("slight downward component", prompt)
        self.assertIn("Reach handle height while still outside", prompt)
        self.assertNotIn("seed 1002", prompt)
        self.assertNotIn("contact_point_id", prompt)
        self.assertNotIn("29.5", prompt)


if __name__ == "__main__":
    unittest.main()
