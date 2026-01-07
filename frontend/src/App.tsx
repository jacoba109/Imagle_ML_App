import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = 'http://127.0.0.1:8000';

type Prompt = {
  id: string;
  url: string;
};

type Candidate = {
  id: string;
  url: string;
  // Optional debug fields (if you choose to expose them temporarily)
  sim?: number;
};

type GameTodayResponse = {
  game_id: string;
  prompt: Prompt;
  candidates: Candidate[];
  // target_id?: string; // dev-only if you still return it
};

type GuessResponse = {
  correct: boolean;
  correct_id: string;
};

const MAX_ATTEMPTS = 6;

// “Close enough” concept without ML scoring (since user is just choosing).
// For now, reveal the answer after a correct guess or when attempts run out.
const REVEAL_ON_CORRECT = true;

const App: React.FC = () => {
  const [game, setGame] = useState<GameTodayResponse | null>(null);
  const [loading, setLoading] = useState<boolean>(true);
  const [guessLoading, setGuessLoading] = useState<boolean>(false);
  const [error, setError] = useState<string | null>(null);

  const [attempt, setAttempt] = useState<number>(0);
  const [selectedId, setSelectedId] = useState<string | null>(null);

  const [status, setStatus] = useState<'playing' | 'win' | 'lose'>('playing');
  const [correctId, setCorrectId] = useState<string | null>(null);

  const remaining = useMemo(() => Math.max(0, MAX_ATTEMPTS - attempt), [attempt]);

  const fetchGame = async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<GameTodayResponse>(`${API_BASE}/game/today`);
      setGame(res.data);
      setAttempt(0);
      setSelectedId(null);
      setStatus('playing');
      setCorrectId(null);
    } catch (e) {
      console.log("error is here:");
      console.log(e);
      setError('Failed to load today’s game.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchGame();
  }, []);

  const submitGuess = async (guessId: string) => {
    if (!game) return;
    if (status !== 'playing') return;

    setGuessLoading(true);
    setError(null);
    setSelectedId(guessId);

    try {
      const res = await axios.post<GuessResponse>(`${API_BASE}/game/guess`, {
        guess_id: guessId,
        game_id: game.game_id, // backend can ignore for now; helpful later
      });

      const isCorrect = res.data.correct;
      const answerId = res.data.correct_id;

      setAttempt((a) => a + 1);

      if (isCorrect) {
        setStatus('win');
        setCorrectId(answerId);
      } else {
        // If out of attempts, reveal and lose
        setCorrectId(answerId);
        // We can only compute “out of attempts” using the *next* attempt value:
        const nextAttempt = attempt + 1;
        if (nextAttempt >= MAX_ATTEMPTS) {
          setStatus('lose');
        }
      }
    } catch (e) {
      setError('Failed to submit guess.');
    } finally {
      setGuessLoading(false);
    }
  };

  // Reveal logic (you can change this later to be “close enough” using some scoring endpoint)
  const shouldReveal = useMemo(() => {
    if (!game) return false;
    if (status === 'win' && REVEAL_ON_CORRECT) return true;
    if (status === 'lose') return true;
    return false;
  }, [game, status]);

  const correctCandidate = useMemo(() => {
    if (!game || !correctId) return null;
    return game.candidates.find((c) => c.id === correctId) || null;
  }, [game, correctId]);

  return (
    <div className="App">
      <header className="header">
        <h1>Imagle</h1>
        <p className="subtitle">
          One of the photos on the right was submitted as a lookalike for the left one. Match the left image to the correct lookalike on the right.
        </p>
      </header>

      {loading && <p className="status">Loading today’s puzzle…</p>}
      {error && <p className="error">{error}</p>}

      {!loading && game && (
        <div className="layout">
          {/* Left: Prompt */}
          <section className="promptCard">
            <h2 className="sectionTitle">Prompt</h2>
            <div className="promptImageWrap">
              <img src={`${API_BASE}${game.prompt.url}`} alt="Prompt" className="promptImage" />
            </div>

            <div className="hud">
              <div className="hudItem">
                <span className="hudLabel">Attempts</span>
                <span className="hudValue">{attempt} / {MAX_ATTEMPTS}</span>
              </div>
              <div className="hudItem">
                <span className="hudLabel">Remaining</span>
                <span className="hudValue">{remaining}</span>
              </div>
              <div className="hudItem">
                <span className="hudLabel">Status</span>
                <span className="hudValue">
                  {status === 'playing' ? 'Playing' : status === 'win' ? 'You win!' : 'Out of attempts'}
                </span>
              </div>
            </div>

            <div className="actions">
              <button className="btn" onClick={fetchGame} disabled={loading || guessLoading}>
                Refresh
              </button>
            </div>

            {shouldReveal && (
              <div className="reveal">
                <h3 className="revealTitle">Answer</h3>
                <p className="revealText">
                  Correct ID: <b>{correctId}</b>
                </p>
                {correctCandidate && (
                  <img
                    src={`${API_BASE}${correctCandidate.url}`}
                    alt="Correct"
                    className="revealImage"
                  />
                )}
              </div>
            )}
          </section>

          {/* Right: Candidate grid */}
          <section className="candidatesCard">
            <h2 className="sectionTitle">Candidates</h2>

            <div className="grid">
              {game.candidates.map((c) => {
                const isSelected = selectedId === c.id;
                const isCorrect = shouldReveal && correctId === c.id;
                const isWrongSelected = status !== 'playing' && selectedId === c.id && correctId !== c.id;

                return (
                  <button
                    key={c.id}
                    className={[
                      'tile',
                      isSelected ? 'tileSelected' : '',
                      isCorrect ? 'tileCorrect' : '',
                      isWrongSelected ? 'tileWrong' : '',
                    ].join(' ')}
                    onClick={() => submitGuess(c.id)}
                    disabled={guessLoading || status !== 'playing'}
                    title={c.id}
                  >
                    <img src={`${API_BASE}${c.url}`} alt={`Candidate ${c.id}`} className="tileImg" />
                    <div className="tileFooter">
                      <span className="tileId">{c.id}</span>
                      {typeof c.sim === 'number' && (
                        <span className="tileMeta">{c.sim.toFixed(3)}</span>
                      )}
                    </div>
                  </button>
                );
              })}
            </div>

            {guessLoading && <p className="status">Checking…</p>}

            {status === 'playing' && selectedId && !guessLoading && (
              <p className="status">
                Selected: <b>{selectedId}</b>
              </p>
            )}

            {status === 'win' && (
              <p className="status success">Nice. You found the match.</p>
            )}
            {status === 'lose' && (
              <p className="status danger">No more attempts — answer revealed.</p>
            )}
          </section>
        </div>
      )}
    </div>
  );
};

export default App;
