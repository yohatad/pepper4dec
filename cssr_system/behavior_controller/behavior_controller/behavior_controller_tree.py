#!/usr/bin/env python3
"""
Tour Guide BehaviorTree Construction
Builds the complete behavior tree for the tour guide robot
"""

import py_trees
from py_trees.composites import Sequence, Selector, Parallel
from py_trees.decorators import Timeout, Repeat, Retry
from py_trees.common import BlackBoxLevel

from .behavior_controller_behaviors import (
    StartOfTree, SetAnimateBehavior, SetOvertAttentionMode,
    PerformIconicGesture, PerformDeicticGesture, SayText, Navigate,
    RetrieveListOfExhibits, SelectExhibit, DescribeExhibitSpeech,
    ResetRobotPose, GetVisitorResponse, PressYesNoDialogue,
    SetSpeechEvent, IsVisitorDiscovered, IsMutualGazeDiscovered,
    IsASREnabled, IsListWithExhibit, IsVisitorResponseYes
)

# ---------------------------------------------------------------------------
# Helper for viewer readability
# ---------------------------------------------------------------------------
def blackbox(b: py_trees.behaviour.Behaviour, level=BlackBoxLevel.DETAIL):
    b.blackbox_level = level
    return b


def create_tour_guide_tree(node):
    """
    Create the complete tour guide behavior tree

    Args:
        node: rclpy Node
    Returns:
        py_trees Behaviour (root)
    """
    root = blackbox(Sequence(name="TourGuide", memory=True), BlackBoxLevel.COMPONENT)

    root.add_children([
        StartOfTree(name="InitializeTour", node=node),

        create_detect_visitor_tree(node),
        create_engage_visitor_tree(node),
        create_query_visitor_response_tree(node),
        create_visit_exhibit_tree(node),

        create_end_tour_tree(node)
    ])

    return root


# I. Detect Visitor -----------------------------------------------------------
def create_detect_visitor_tree(node):
    detect = blackbox(Sequence(name="DetectVisitor", memory=True))

    setup = Parallel(
        name="SetupDetection",
        policy=py_trees.common.ParallelPolicy.SuccessOnAll(synchronise=False)
    )
    setup.add_children([
        SetAnimateBehavior(name="EnableAnimation", node=node, animate_enabled=True),
        SetOvertAttentionMode(name="SetScanning", node=node, mode="scanning"),
    ])

    wait_visitor = Timeout(
        name="WaitForVisitor(30s)",
        child=IsVisitorDiscovered(name="WaitForVisitor", node=node, timeout=30.0),
        duration=30.0
    )

    detect.add_children([setup, wait_visitor])
    return detect


# II. Engage Visitor ----------------------------------------------------------
def create_engage_visitor_tree(node):
    engage = blackbox(Sequence(name="EngageVisitor", memory=True))

    welcome = blackbox(Sequence(name="WelcomeGesture", memory=True))
    welcome.add_children([
        SetAnimateBehavior(name="DisableAnimForGesture", node=node, animate_enabled=False),
        PerformIconicGesture(name="Wave", node=node, gesture_type="welcome", duration=2.0),
        SetAnimateBehavior(name="EnableAnimAfterGesture", node=node, animate_enabled=True),
    ])

    engage.add_children([
        welcome,
        SayText(name="WelcomeSpeech", node=node, text_id="welcome_speech", duration=3.0),
        SetOvertAttentionMode(name="SeekGaze", node=node, mode="seeking"),
        Timeout(
            name="SeekGaze(10s)",
            child=IsMutualGazeDiscovered(name="EstablishGaze", node=node, timeout=10.0),
            duration=10.0
        ),
        SetOvertAttentionMode(name="SocialMode", node=node, mode="social"),
        SayText(name="QueryTour", node=node, text_id="query_tour_speech", duration=3.0),
    ])
    return engage


# III. Query Visitor Response -------------------------------------------------
def create_query_visitor_response_tree(node):
    query = blackbox(Sequence(name="QueryVisitorResponse", memory=True))

    # Try ASR first, then button fallback
    get_response = Selector(name="GetResponse", memory=False)

    asr_path = Sequence(name="ASRResponse", memory=True)
    asr_path.add_children([
        IsASREnabled(name="CheckASR", node=node),
        SetSpeechEvent(name="EnableASR", node=node, event_enabled=True),
        Timeout(
            name="ASR(10s)",
            child=GetVisitorResponse(name="ListenForResponse", node=node, timeout=10.0),
            duration=10.0
        ),
        SetSpeechEvent(name="DisableASR", node=node, event_enabled=False),
    ])

    button_seq = Sequence(name="ButtonSequence", memory=True, children=[
        SayText(name="PressButtonPrompt", node=node, text_id="press_yes_no_speech", duration=2.0),
        Timeout(
            name="Buttons(15s)",
            child=PressYesNoDialogue(name="WaitForButton", node=node, timeout=15.0),
            duration=15.0
        )
    ])
    button_path = Retry(name="ButtonRetry", child=button_seq, num_failures=3)

    get_response.add_children([asr_path, button_path])

    # Process response
    process = Selector(name="ProcessResponse", memory=False)

    reject = Sequence(name="HandleRejection", memory=True)
    reject.add_children([
        SayText(name="MaybeAnotherTime", node=node, text_id="maybe_another_time_speech", duration=2.0),
        py_trees.behaviours.Failure(name="TourDeclined")
    ])

    process.add_children([
        IsVisitorResponseYes(name="CheckYes", node=node),  # SUCCESS -> continue
        reject                                              # else -> fail
    ])

    query.add_children([get_response, process])
    return query


# IV. Visit Exhibits ----------------------------------------------------------
def create_visit_exhibit_tree(node):
    visit = blackbox(Sequence(name="VisitExhibit", memory=True))

    visit.add_children([
        RetrieveListOfExhibits(name="GetExhibits", node=node),
        ResetRobotPose(name="ResetPose", node=node),
        create_visit_loop(node)
    ])
    return visit


def create_visit_loop(node):
    """
    Visit exhibits until the list is empty.
    Repeat the 'step' while it returns SUCCESS; when empty, Repeat fails,
    so a Selector falls through to a dummy Success to finish the subtree cleanly.
    """
    step = Sequence(name="VisitOneIfAny", memory=True)
    step.add_children([
        IsListWithExhibit(name="HasNext", node=node),
        create_single_exhibit_visit(node)
    ])

    loop = Repeat(name="ExhibitLoop", num_success=-1, child=step)  # repeat while step succeeds
    done = py_trees.behaviours.Success(name="NoMoreExhibits")

    return Selector(name="LoopOrDone", memory=False, children=[loop, done])


def create_single_exhibit_visit(node):
    single = blackbox(Sequence(name="SingleExhibitVisit", memory=True))
    single.add_children([
        SelectExhibit(name="ChooseExhibit", node=node, selection_strategy="sequential"),
        SayText(name="FollowMe", node=node, text_id="follow_me_speech", duration=2.0),
        create_navigate_to_location_tree(node),
        create_present_exhibit_tree(node)
    ])
    return single


def create_navigate_to_location_tree(node):
    nav = blackbox(Sequence(name="NavigateToLocation", memory=True))
    nav.add_children([
        SetOvertAttentionMode(name="DisableAttention", node=node, mode="disabled"),
        SetAnimateBehavior(name="DisableAnimForNav", node=node, animate_enabled=False),
        Timeout(
            name="Navigate(60s)",
            child=Navigate(name="NavigateToExhibit", node=node, timeout=60.0),
            duration=60.0
        ),
        SetOvertAttentionMode(name="SeekGazeAtExhibit", node=node, mode="seeking"),
        Timeout(
            name="SeekGaze(10s)",
            child=IsMutualGazeDiscovered(name="CheckGazeAtExhibit", node=node, timeout=10.0),
            duration=10.0
        )
    ])
    return nav


def create_present_exhibit_tree(node):
    present = blackbox(Sequence(name="PresentExhibit", memory=True))
    present.add_children([
        SetOvertAttentionMode(name="SocialForIntro", node=node, mode="social"),
        DescribeExhibitSpeech(name="IntroSpeech", node=node, speech_part="introduction", duration=4.0),

        SetOvertAttentionMode(name="LookAtExhibit", node=node, mode="location"),
        PerformDeicticGesture(name="PointAtExhibit", node=node, duration=2.0),

        SetOvertAttentionMode(name="SocialForDetails", node=node, mode="social"),
        DescribeExhibitSpeech(name="DetailSpeech", node=node, speech_part="details", duration=4.0)
    ])
    return present


# V. End Tour -----------------------------------------------------------------
def create_end_tour_tree(node):
    end_seq = blackbox(Sequence(name="EndTour", memory=True))
    end_seq.add_children([
        SayText(name="GoodbyeSpeech", node=node, text_id="goodbye_speech", duration=3.0),
        PerformIconicGesture(name="GoodbyeWave", node=node, gesture_type="goodbye", duration=2.0)
    ])
    return end_seq


def display_tree(root: py_trees.behaviour.Behaviour):
    """Pretty-print the tree structure."""
    print("\n" + "="*80)
    print("TOUR GUIDE BEHAVIOR TREE STRUCTURE")
    print("="*80)
    print(py_trees.display.unicode_tree(root, show_status=True))
    print("="*80 + "\n")
