#!/usr/bin/env python3
"""
Cleanup Elasticsearch Indices

This script deletes all Elasticsearch indices that start with 'parlant'.
Useful for cleaning up after testing or resetting the database.

Usage:
    python scripts/cleanup_elasticsearch.py

Environment Variables:
    ELASTICSEARCH__HOST: Elasticsearch host (default: localhost)
    ELASTICSEARCH__PORT: Elasticsearch port (default: 9200)
    ELASTICSEARCH__USERNAME: Elasticsearch username (optional)
    ELASTICSEARCH__PASSWORD: Elasticsearch password (optional)
"""

import asyncio
import sys
from typing import Any

try:
    from elasticsearch import AsyncElasticsearch
except ImportError:
    print("Error: elasticsearch package is required.")
    print("Install it with: pip install elasticsearch")
    sys.exit(1)


async def cleanup_elasticsearch_indices(skip_confirm: bool = False) -> None:
    """Delete all parlant* indices from Elasticsearch."""

    # Import the helper function to create client from env
    from parlant.adapters.db.elasticsearch import create_elasticsearch_document_client_from_env

    print("Connecting to Elasticsearch...")
    client = create_elasticsearch_document_client_from_env()

    try:
        # Get cluster info
        info: dict[str, Any] = await client.info()
        print(f"✓ Connected to Elasticsearch {info['version']['number']}")
        print(f"  Cluster: {info['cluster_name']}")
        print()

        # List all indices matching parlant*
        print("Searching for parlant* indices...")
        indices_response: list[dict[str, Any]] = await client.cat.indices(
            index="parlant*", format="json", h="index,docs.count,store.size"
        )

        if not indices_response:
            print("✓ No parlant* indices found. Nothing to clean up.")
            return

        print(f"Found {len(indices_response)} indices to delete:")
        print()

        # Show what will be deleted
        for idx_info in indices_response:
            index_name: str = idx_info["index"]
            doc_count: str = idx_info.get("docs.count", "0")
            size: str = idx_info.get("store.size", "0b")
            print(f"  • {index_name}")
            print(f"    Documents: {doc_count}, Size: {size}")

        print()

        # Confirm deletion
        if not skip_confirm:
            response = input("Do you want to delete these indices? (yes/no): ")
            if response.lower() not in ["yes", "y"]:
                print("Cancelled.")
                return
        else:
            print("Skipping confirmation (--yes flag provided)")

        print()
        print("Deleting indices...")

        # Delete each index
        deleted_count = 0
        failed_count = 0

        for idx_info in indices_response:
            index_name: str = idx_info["index"]
            try:
                await client.indices.delete(index=index_name)
                print(f"  ✓ Deleted: {index_name}")
                deleted_count += 1
            except Exception as e:
                print(f"  ✗ Failed to delete {index_name}: {e}")
                failed_count += 1

        print()
        print("=" * 60)
        print(f"Cleanup complete:")
        print(f"  • Deleted: {deleted_count} indices")
        if failed_count > 0:
            print(f"  • Failed: {failed_count} indices")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.close()


async def list_indices_only() -> None:
    """List all parlant* indices without deleting."""
    from parlant.adapters.db.elasticsearch import create_elasticsearch_document_client_from_env

    print("Connecting to Elasticsearch...")
    client = create_elasticsearch_document_client_from_env()

    try:
        # Get cluster info
        info: dict[str, Any] = await client.info()
        print(f"✓ Connected to Elasticsearch {info['version']['number']}")
        print()

        # List all indices matching parlant*
        print("Parlant indices:")
        print()

        indices_response: list[dict[str, Any]] = await client.cat.indices(
            index="parlant*", format="json", h="index,docs.count,store.size,health"
        )

        if not indices_response:
            print("  No parlant* indices found.")
            return

        for idx_info in indices_response:
            index_name: str = idx_info["index"]
            doc_count: str = idx_info.get("docs.count", "0")
            size: str = idx_info.get("store.size", "0b")
            health: str = idx_info.get("health", "unknown")

            health_symbol = "✓" if health == "green" else "⚠" if health == "yellow" else "✗"
            print(f"  {health_symbol} {index_name}")
            print(f"    Documents: {doc_count}, Size: {size}, Health: {health}")

        print()
        print(f"Total: {len(indices_response)} indices")

    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)
    finally:
        await client.close()


def main() -> None:
    """Main entry point."""
    skip_confirm = "--yes" in sys.argv or "-y" in sys.argv

    if len(sys.argv) > 1 and (
        sys.argv[1] in ["--list", "-l"] or any(arg in ["--list", "-l"] for arg in sys.argv)
    ):
        print("Listing parlant* indices...")
        print()
        asyncio.run(list_indices_only())
    else:
        asyncio.run(cleanup_elasticsearch_indices(skip_confirm=skip_confirm))


if __name__ == "__main__":
    main()
