import numpy as np

import logger
import music
from markov_chain_model import MarkovChainList
from midi2img import midi2image


def single_use_function():
    # Get the music input vector time series
    voice = 3
    original_score = music.load_midi("unfin.mid")
    score_vector_ts = music.to_vector_ts(original_score)

    mcl = MarkovChainList()

    mcl.fit(score_vector_ts)

    # predicted states to numpy array
    predicted_states = mcl.model[voice].generate_states(score_vector_ts[0, voice], 128)
    predicted_states = np.array([predicted_states]).T

    predicted = music.from_vector_ts(predicted_states)
    fp = predicted.write("midi", fp="markov_chain_output_voice_3.mid")

    midi2image(
        "markov_chain_output_voice_3.mid", "markov_chain_output_voice_3.png", 0, 128
    )


def main():
    single_use_function()


if __name__ == "__main__":
    log = logger.setup_logger(__name__)
    log.info("Starting...")

    # Run the main
    main()
