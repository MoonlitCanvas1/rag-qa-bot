"""Typer command-line interface: ingest, query, chat, reset."""

import logging
from pathlib import Path
from typing import Optional

import typer

from rag_bot.config import Settings, configure_logging, get_settings
from rag_bot.generate import Answer, GenerationError
from rag_bot.ingest import IngestError
from rag_bot.pipeline import RAGPipeline

logger = logging.getLogger(__name__)

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Ask questions about your documents (RAG with ChromaDB + Claude).",
)

EXIT_WORDS = {"exit", "quit", ":q"}


@app.callback()
def main(
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable debug logging."),
) -> None:
    settings = get_settings()
    configure_logging("DEBUG" if verbose else settings.log_level)


def _pipeline(settings: Settings | None = None) -> RAGPipeline:
    return RAGPipeline(settings or get_settings())


def format_answer(answer: Answer) -> str:
    """Render an answer plus its citations as plain text."""
    lines = [answer.text]
    if answer.citations:
        lines += ["", "Sources:"]
        for c in answer.citations:
            page = f", p. {c.page}" if c.page else ""
            lines.append(f"  [{c.index}] {c.source}{page} (score {c.score:.2f})")
    return "\n".join(lines)


@app.command()
def ingest(
    path: Optional[Path] = typer.Argument(
        None, help="File or directory to ingest. Defaults to DATA_DIR."
    ),
) -> None:
    """Load, chunk, embed, and store documents (.txt, .md, .pdf, .docx)."""
    try:
        stats = _pipeline().ingest(path)
    except IngestError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(
        f"Ingested {stats.files} file(s) into {stats.chunks} chunk(s); skipped {stats.skipped}."
    )


@app.command()
def query(
    question: str = typer.Argument(..., help="Your question."),
    top_k: Optional[int] = typer.Option(None, "--top-k", "-k", min=1, help="Override TOP_K."),
    min_score: Optional[float] = typer.Option(
        None, "--min-score", min=-1.0, max=1.0, help="Override MIN_SCORE."
    ),
) -> None:
    """Ask a single question and print the answer with sources."""
    try:
        answer = _pipeline().ask(question, top_k=top_k, min_score=min_score)
    except GenerationError as exc:
        logger.error("%s", exc)
        raise typer.Exit(code=1) from exc
    typer.echo(format_answer(answer))


@app.command()
def chat() -> None:
    """Interactive Q&A. Each question is answered independently (no chat memory)."""
    pipeline = _pipeline()
    typer.echo("RAG chat. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            question = typer.prompt("you", default="", show_default=False, prompt_suffix=" > ")
        except (EOFError, typer.Abort):
            typer.echo("")
            break
        question = question.strip()
        if not question:
            continue
        if question.lower() in EXIT_WORDS:
            break
        try:
            answer = pipeline.ask(question)
        except GenerationError as exc:
            logger.error("%s", exc)
            continue
        typer.echo(f"\n{format_answer(answer)}\n")


@app.command()
def reset(
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip the confirmation prompt."),
) -> None:
    """Delete all ingested data from the vector store."""
    settings = get_settings()
    if not yes:
        typer.confirm(f"Delete collection {settings.collection_name!r} in {settings.chroma_dir}?", abort=True)
    _pipeline(settings).reset()
    typer.echo("Vector store reset.")


if __name__ == "__main__":
    app()
