import argparse
import asyncio
import json
import logging
import os

import cv2
import numpy as np
import tqdm
from google import genai
from google.auth import default
from google.genai import types
from PIL import Image

from build_dataset.task_specification import TASK_SPECIFICATIONS

LOCATION = "us-central1"
PROJECT = ""  # replace
MODEL = "gemini-2.5-flash-preview-04-17"
EPISODE_LENGTH = 300


def get_gemini_client():
    """Initialize and return a Gemini client using environment-based authentication."""
    try:
        credentials, _ = default()
        return genai.Client(
            vertexai=True, project=PROJECT, location=LOCATION, credentials=credentials
        )
    except Exception as e:
        logging.error(f"Failed to initialize Gemini client: {e}")
        raise


client = get_gemini_client()


async def query_keypoints_async(
    img: Image.Image, query: str, i: int, pbar: tqdm.tqdm
) -> str:
    """Query Gemini for keypoint detection asynchronously."""
    loop = asyncio.get_event_loop()
    contents = [img, query]

    try:
        resp = await asyncio.wait_for(
            loop.run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=MODEL,
                    contents=contents,
                    config=types.GenerateContentConfig(temperature=0.5),
                ),
            ),
            timeout=120,
        )
        pbar.update(1)
        return resp.text
    except asyncio.TimeoutError:
        logging.warning(f"Keypoint query for sample {i} timed out after 6 minutes")
        pbar.update(1)
        return None
    except Exception as e:
        logging.warning(f"Failed to get keypoint response for sample {i}: {e}")
        pbar.update(1)
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate dataset for a specific task")
    parser.add_argument(
        "--task",
        choices=list(TASK_SPECIFICATIONS.keys()),
        required=True,
        help="Task to generate dataset for",
    )
    parser.add_argument(
        "--method",
        choices=["gemini"],
        default="gemini",
        help="Method to use for trajectory generation",
    )
    parser.add_argument(
        "--split", default="train1", help="Split to generate dataset for"
    )
    parser.add_argument("--img", help="Split to generate dataset for")
    args = parser.parse_args()

    task_spec = TASK_SPECIFICATIONS[args.task].create()

    img = Image.open(args.img)
    # Create necessary directories
    folder = f"data/real/{args.task}/{args.split}"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(os.path.join(folder, "visualizations", "keypoints"), exist_ok=True)

    img_path = os.path.join(folder, "original_image.png")
    img.save(img_path)
    logging.info(f"Saved original image to {img_path}")

    query = f"""{task_spec.keypoint_task}
The answer should follow the json format: [{', '.join([f'{{"name": "{name}", "point": [...]}}' for name in task_spec.item_names])}]
The points are in [y, x] format normalized to 0-1000."""

    logging.info("Querying Gemini for keypoint detection...")
    with tqdm.tqdm(total=1, desc="Getting keypoint responses") as pbar:

        async def main():
            tasks = [query_keypoints_async(img, query, i, pbar) for i in range(1)]
            keypoint_responses = await asyncio.gather(*tasks)
            return keypoint_responses

        keypoint_responses = asyncio.run(main())

    responses = [
        response.strip("'").strip("`json\n").strip("`")
        for response in keypoint_responses
    ]
    responses = [json.loads(response) for response in responses]
    for i, response in enumerate(responses):
        img_cv = np.array(img)
        img_with_points = img_cv.copy()
        for item in response:
            name = item["name"]
            item["point"][0] = int(item["point"][0] / 1000 * img.height)
            item["point"][1] = int(item["point"][1] / 1000 * img.width)
            x = item["point"][1]
            y = item["point"][0]
            cv2.circle(img_with_points, (x, y), 5, (0, 0, 255), -1)
            cv2.putText(
                img_with_points,
                name,
                (x + 10, y),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 0, 0),
                1,
            )

        img_with_points_pil = Image.fromarray(img_with_points)
        img_with_points_pil.save(
            os.path.join(
                folder,
                "visualizations",
                "keypoints",
                f"initial_with_points_query_{i}.png",
            )
        )

        # Save the keypoint responses as JSON
        keypoints_folder = os.path.join(folder, "responses")
        os.makedirs(keypoints_folder, exist_ok=True)

        with open(
            os.path.join(keypoints_folder, "all_keypoint_responses.json"), "w"
        ) as f:
            json.dump(responses[0], f, indent=2)

        logging.info(f"Saved keypoint responses to {keypoints_folder}")
