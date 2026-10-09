"""Built-in Benchmark Tasks for Multi-Harness Auditing."""

from __future__ import annotations

from polyharness.schema.adp import (
    EnvironmentSpec,
    Observation,
    ObservationType,
    Step,
    ToolCall,
    ToolDefinition,
    ToolParameter,
    ToolResult,
    Trajectory,
    TrajectoryMetadata,
)


def get_standard_benchmarks() -> list[Trajectory]:
    """Returns a curated suite of standard ADP benchmark tasks."""
    # 1. Web E-Commerce Search & Filter Task
    web_tools = [
        ToolDefinition(
            name="search_products",
            description="Query the product catalog with keywords and filters",
            parameters={
                "query": ToolParameter(name="query", type="string", description="Search terms", required=True),
                "category": ToolParameter(name="category", type="string", description="Category filter"),
                "max_price": ToolParameter(name="max_price", type="number", description="Maximum price in USD"),
            },
        ),
        ToolDefinition(
            name="add_to_cart",
            description="Add item to user shopping cart",
            parameters={
                "product_id": ToolParameter(name="product_id", type="string", description="ID of product", required=True),
                "quantity": ToolParameter(name="quantity", type="integer", description="Units to add", default=1),
            },
        ),
    ]

    traj_web = Trajectory(
        id="bench_ecommerce_01",
        task="Find a wireless noise-cancelling headphone under $200 and add 1 unit of the highest rated item to cart.",
        system_prompt="You are an autonomous shopping assistant. Use available tools to fulfill user requirements.",
        environment=EnvironmentSpec(name="ecommerce_sandbox", harness_id="adp_bench"),
        tools=web_tools,
        steps=[
            Step(
                step_index=0,
                thought="I need to search for wireless noise-cancelling headphones priced under $200.",
                tool_calls=[
                    ToolCall(
                        name="search_products",
                        arguments={"query": "wireless noise-cancelling headphones", "max_price": 200},
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_call_id="",
                        name="search_products",
                        content='[{"id": "PROD-882", "name": "AcousticPro ANC", "price": 179.99, "rating": 4.8}, {"id": "PROD-104", "name": "SoundWave Lite", "price": 89.99, "rating": 4.1}]',
                    )
                ],
                observation=Observation(
                    primary_type=ObservationType.HTML,
                    raw_content="<div class='product-list'><div class='card' id='PROD-882'><h3>AcousticPro ANC</h3><span class='price'>$179.99</span><span class='rating'>4.8 stars</span><button id='btn-add-882'>Add to Cart</button></div></div>",
                    html_content="<div class='product-list'><div class='card' id='PROD-882'><h3>AcousticPro ANC</h3><span class='price'>$179.99</span><span class='rating'>4.8 stars</span><button id='btn-add-882'>Add to Cart</button></div></div>",
                    markdown="# Search Results\n- **AcousticPro ANC** ($179.99, Rating: 4.8) [Button: Add to Cart]",
                    accessibility_tree="RootWebArea\n  heading 'Search Results'\n  StaticText 'AcousticPro ANC - $179.99 (4.8 stars)'\n  button 'Add to Cart' [id=btn-add-882]",
                ),
            ),
            Step(
                step_index=1,
                thought="AcousticPro ANC (PROD-882) is the highest rated at 4.8 and costs $179.99. Now adding to cart.",
                tool_calls=[
                    ToolCall(
                        name="add_to_cart",
                        arguments={"product_id": "PROD-882", "quantity": 1},
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_call_id="",
                        name="add_to_cart",
                        content='{"status": "success", "cart_item_count": 1, "total": 179.99}',
                    )
                ],
            ),
        ],
        outcome={"success": True, "final_answer": "Successfully added AcousticPro ANC ($179.99, 4.8 stars) to cart."},
        metadata=TrajectoryMetadata(tags=["benchmark", "ecommerce"]),
    )

    # 2. SQL Data Analytics Task
    sql_tools = [
        ToolDefinition(
            name="execute_sql",
            description="Run analytical SQL query on PostgreSQL database",
            parameters={
                "query": ToolParameter(name="query", type="string", description="SQL SELECT query", required=True),
            },
        ),
        ToolDefinition(
            name="describe_table",
            description="Inspect schema columns and types for a table",
            parameters={
                "table_name": ToolParameter(name="table_name", type="string", description="Table name", required=True),
            },
        ),
    ]

    traj_sql = Trajectory(
        id="bench_sql_analytics_02",
        task="Calculate the monthly gross revenue for Q3 2024 from the transactions table.",
        system_prompt="You are a senior data engineer assistant. Verify schemas before running aggregations.",
        environment=EnvironmentSpec(name="analytics_postgres", harness_id="adp_bench"),
        tools=sql_tools,
        steps=[
            Step(
                step_index=0,
                thought="First I should inspect the schema of the transactions table to identify amount and date columns.",
                tool_calls=[
                    ToolCall(name="describe_table", arguments={"table_name": "transactions"})
                ],
                tool_results=[
                    ToolResult(
                        tool_call_id="",
                        name="describe_table",
                        content='{"columns": [{"name": "id", "type": "uuid"}, {"name": "created_at", "type": "timestamp"}, {"name": "amount_cents", "type": "integer"}, {"name": "status", "type": "varchar"}]}',
                    )
                ],
            ),
            Step(
                step_index=1,
                thought="Amounts are stored in amount_cents. I need to sum amount_cents / 100.0 grouped by month for 2024-07-01 to 2024-09-30 where status = 'settled'.",
                tool_calls=[
                    ToolCall(
                        name="execute_sql",
                        arguments={
                            "query": "SELECT to_char(created_at, 'YYYY-MM') as month, SUM(amount_cents)/100.0 as revenue FROM transactions WHERE created_at >= '2024-07-01' AND created_at < '2024-10-01' AND status = 'settled' GROUP BY 1 ORDER BY 1;"
                        },
                    )
                ],
                tool_results=[
                    ToolResult(
                        tool_call_id="",
                        name="execute_sql",
                        content='[{"month": "2024-07", "revenue": 142500.00}, {"month": "2024-08", "revenue": 158200.00}, {"month": "2024-09", "revenue": 169800.00}]',
                    )
                ],
            ),
        ],
        outcome={
            "success": True,
            "final_answer": "Q3 2024 Monthly Gross Revenue:\n- July: $142,500.00\n- August: $158,200.00\n- September: $169,800.00\nTotal Q3: $470,500.00",
        },
        metadata=TrajectoryMetadata(tags=["benchmark", "sql"]),
    )

    return [traj_web, traj_sql]
