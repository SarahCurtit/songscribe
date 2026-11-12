"""Local web viewer: listen to a recording and compare what the models heard.

Chords land on a timeline beside the chart's own; a melody lands on a staff on
that same timeline, so the written line and the transcribed one can be read
against each other. The code is generic; the songs are not. Recordings are
passed in at runtime (``songscribe-viewer --examples PATH``, ``--audio FILE``)
and are not copied anywhere, so private or licensed audio stays where it is.
"""
