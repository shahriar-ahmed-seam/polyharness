"""Observation Representation Transformer.

Transforms observations across HTML, Accessibility Tree, Markdown, and JSON.
Prevents models from overfitting to a specific DOM or text serialization.
"""

from __future__ import annotations

import copy

from bs4 import BeautifulSoup

from polyharness.schema.adp import ObservationType, Trajectory


class ObservationTransformer:
    """Transforms web/text observations into alternate structural representations."""

    @staticmethod
    def html_to_markdown(html_content: str) -> str:
        """Convert raw HTML snippet to clean semantic markdown."""
        soup = BeautifulSoup(html_content, "html.parser")

        # Strip scripts and styles
        for tag in soup(["script", "style", "noscript", "svg"]):
            tag.decompose()

        lines = []
        for elem in soup.find_all(["h1", "h2", "h3", "p", "a", "button", "input", "li"]):
            text = elem.get_text(strip=True)
            if not text and elem.name not in ["input", "button"]:
                continue

            if elem.name == "h1":
                lines.append(f"# {text}")
            elif elem.name == "h2":
                lines.append(f"## {text}")
            elif elem.name == "h3":
                lines.append(f"### {text}")
            elif elem.name == "a":
                href = elem.get("href", "#")
                lines.append(f"[{text}]({href})")
            elif elem.name == "button":
                lines.append(f"[Button: {text or elem.get('id', 'submit')}]")
            elif elem.name == "input":
                placeholder = elem.get("placeholder", "")
                val = elem.get("value", "")
                lines.append(f"[Input: {placeholder or val or elem.get('name', 'text')}]")
            elif elem.name == "li":
                lines.append(f"- {text}")
            elif elem.name == "p":
                lines.append(text)

        result = "\n".join(lines)
        return result if result.strip() else soup.get_text(separator="\n", strip=True)

    @staticmethod
    def html_to_accessibility_tree(html_content: str) -> str:
        """Convert raw HTML snippet to an AXTree format (used by BrowserGym/Playwright)."""
        soup = BeautifulSoup(html_content, "html.parser")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()

        tree_lines = []
        node_id = 1

        def traverse(node, depth=0):
            nonlocal node_id
            if node.name is None:
                text = str(node).strip()
                if text:
                    tree_lines.append(f"{'  ' * depth}StaticText [id={node_id}] '{text}'")
                    node_id += 1
                return

            role = node.name
            if role == "a":
                role = "link"
            elif role == "button":
                role = "button"
            elif role == "input":
                role = f"textbox (name='{node.get('name', '')}')"
            elif role in ["h1", "h2", "h3"]:
                role = f"heading ({role})"

            aria_label = node.get("aria-label") or node.get("id") or ""
            label_suffix = f" '{aria_label}'" if aria_label else ""

            # Only print relevant interactive or structural elements
            is_interactive_or_structural = (
                role in ["link", "button", "body", "form", "main"]
                or role.startswith("heading")
                or role.startswith("textbox")
                or bool(aria_label)
            )
            if is_interactive_or_structural:
                tree_lines.append(f"{'  ' * depth}{role} [id={node_id}]{label_suffix}")
                node_id += 1

            for child in node.children:
                traverse(child, depth + 1)

        traverse(soup)
        return "\n".join(tree_lines) if tree_lines else "RootWebArea [id=0]"

    def populate_multi_representations(self, trajectory: Trajectory) -> Trajectory:
        """Enrich all observations in a trajectory with both AXTree, Markdown, and HTML."""
        enriched = copy.deepcopy(trajectory)
        for step in enriched.steps:
            if step.observation:
                obs = step.observation
                content = obs.raw_content

                # Detect if content is HTML
                is_html = ("<html" in content.lower()) or ("<div" in content.lower()) or ("<body" in content.lower())

                if is_html:
                    if not obs.html_content:
                        obs.html_content = content
                    if not obs.markdown:
                        obs.markdown = self.html_to_markdown(content)
                    if not obs.accessibility_tree:
                        obs.accessibility_tree = self.html_to_accessibility_tree(content)
                else:
                    if not obs.markdown:
                        obs.markdown = content

                obs.primary_type = ObservationType.MULTI_REPRESENTATION

        return enriched

    def synthesize_format_variant(
        self, trajectory: Trajectory, target_format: ObservationType
    ) -> Trajectory:
        """Create a new trajectory where observations are swapped to the target format."""
        variant = self.populate_multi_representations(trajectory)
        variant.id = f"{trajectory.id}_synth_obs_{target_format.value}"

        for step in variant.steps:
            if step.observation:
                obs = step.observation
                if target_format == ObservationType.MARKDOWN and obs.markdown:
                    obs.raw_content = obs.markdown
                    obs.primary_type = ObservationType.MARKDOWN
                elif target_format == ObservationType.ACCESSIBILITY_TREE and obs.accessibility_tree:
                    obs.raw_content = obs.accessibility_tree
                    obs.primary_type = ObservationType.ACCESSIBILITY_TREE
                elif target_format == ObservationType.HTML and obs.html_content:
                    obs.raw_content = obs.html_content
                    obs.primary_type = ObservationType.HTML

        variant.metadata.notes = f"Observation synthesized for format {target_format.value}"
        return variant
