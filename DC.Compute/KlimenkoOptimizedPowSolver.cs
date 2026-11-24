using System.Buffers.Text;
using System.Diagnostics;
using System.Globalization;
using System.Security.Cryptography;
using System.Text;

namespace DC.Compute;

public sealed class KlimenkoOptimizedPowSolver : IPowSolver
{
    private const int ProgressBatchSize = 16_384;
    private const ulong CancellationCheckMask = 1_023;

    public Task<PowResult> SolveAsync(
        string seed,
        int difficultyBits,
        int workerCount,
        IProgress<PowProgress>? progress,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry = null)
    {
        ValidateArguments(seed, difficultyBits, workerCount);

        return Task.Run(
            () => SolveCore(seed, difficultyBits, workerCount, progress, cancellationToken, telemetry),
            CancellationToken.None);
    }

    private static void ValidateArguments(string seed, int difficultyBits, int workerCount)
    {
        if (seed is null || seed.Length != 6 || seed.Any(c => c is < '0' or > '9'))
            throw new ArgumentException(
                "The seed must contain exactly six decimal digits (0–9).", nameof(seed));

        if (difficultyBits is < 1 or > PowSolver.MaximumDifficultyBits)
            throw new ArgumentOutOfRangeException(
                nameof(difficultyBits), "Difficulty must be between 1 and 24 bits.");

        if (workerCount is < 1 or > PowSolver.MaximumWorkers)
            throw new ArgumentOutOfRangeException(
                nameof(workerCount), "Worker count must be between 1 and 100.");
    }

    private static PowResult SolveCore(
        string seed,
        int difficultyBits,
        int workerCount,
        IProgress<PowProgress>? progress,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry)
    {
        var solutions = new PowSolution[PowSolver.TaskCount];
        int nextTask = -1;
        int completed = 0;
        long totalAttempts = 0;
        var elapsed = Stopwatch.StartNew();

        var options = new ParallelOptions
        {
            MaxDegreeOfParallelism = workerCount,
            CancellationToken = cancellationToken
        };

        telemetry?.Start();

        try
        {
            Parallel.For(0, workerCount, options, workerId =>
            {
                using SHA256 sha256 = SHA256.Create();

                while (true)
                {
                    cancellationToken.ThrowIfCancellationRequested();

                    int taskId = Interlocked.Increment(ref nextTask);
                    if (taskId >= PowSolver.TaskCount)
                        break;

                    int lane = telemetry?.BeginWork(taskId, workerId) ?? -1;

                    try
                    {
                        progress?.Report(new PowProgress(
                            taskId,
                            null,
                            Volatile.Read(ref completed),
                            Interlocked.Read(ref totalAttempts)));

                        (ulong nonce, long attempts) = SolveChallenge(
                            seed,
                            taskId,
                            difficultyBits,
                            sha256,
                            cancellationToken,
                            telemetry,
                            lane);

                        cancellationToken.ThrowIfCancellationRequested();

                        var solution = new PowSolution(taskId, nonce);
                        solutions[taskId] = solution;

                        telemetry?.Solved(taskId);

                        long currentAttempts = Interlocked.Add(ref totalAttempts, attempts);
                        int currentCompleted = Interlocked.Increment(ref completed);

                        progress?.Report(new PowProgress(
                            taskId,
                            solution,
                            currentCompleted,
                            currentAttempts));
                    }
                    finally
                    {
                        telemetry?.EndWork(lane);
                    }
                }
            });

            cancellationToken.ThrowIfCancellationRequested();
            elapsed.Stop();

            return new PowResult(
                Array.AsReadOnly(solutions),
                elapsed.Elapsed,
                completed,
                workerCount,
                Interlocked.Read(ref totalAttempts));
        }
        finally
        {
            elapsed.Stop();
            telemetry?.Finish();
        }
    }

    private static (ulong Nonce, long Attempts) SolveChallenge(
        string seed,
        int taskId,
        int difficultyBits,
        SHA256 sha256,
        CancellationToken cancellationToken,
        PowTelemetry? telemetry,
        int lane)
    {
        byte[] prefix = Encoding.UTF8.GetBytes(
            string.Create(
                CultureInfo.InvariantCulture,
                $"BSU-POW-1|{seed}|{taskId}|") );

        Span<byte> message = stackalloc byte[prefix.Length + 20];
        prefix.AsSpan().CopyTo(message);

        Span<byte> hash = stackalloc byte[32];
        long attempts = 0;
        long reportedAttempts = 0;

        try
        {
            for (ulong nonce = 0; ; nonce++)
            {
                if ((nonce & CancellationCheckMask) == 0)
                    cancellationToken.ThrowIfCancellationRequested();

                if (!Utf8Formatter.TryFormat(nonce, message[prefix.Length..], out int digits))
                    throw new InvalidOperationException("Could not encode the nonce.");

                // Reuse the worker's SHA256 instance instead of invoking the static
                // one-shot SHA256.HashData API for every candidate nonce.
                if (!sha256.TryComputeHash(message[..(prefix.Length + digits)], hash, out int bytesWritten)
                    || bytesWritten != 32)
                {
                    throw new CryptographicException("SHA-256 did not produce a 32-byte hash.");
                }

                attempts++;

                if (telemetry is not null && attempts - reportedAttempts >= ProgressBatchSize)
                {
                    telemetry.AddAttempts(lane, taskId, attempts - reportedAttempts);
                    reportedAttempts = attempts;
                }

                if (HasLeadingZeroBits(hash, difficultyBits))
                    return (nonce, attempts);

                if (nonce == ulong.MaxValue)
                    throw new InvalidOperationException(
                        $"Task {taskId} exhausted all unsigned 64-bit nonces.");
            }
        }
        finally
        {
            telemetry?.AddAttempts(lane, taskId, attempts - reportedAttempts);
        }
    }

    private static bool HasLeadingZeroBits(ReadOnlySpan<byte> hash, int difficultyBits)
    {
        int fullBytes = difficultyBits / 8;

        for (int i = 0; i < fullBytes; i++)
        {
            if (hash[i] != 0)
                return false;
        }

        int remainingBits = difficultyBits % 8;
        return remainingBits == 0 || (hash[fullBytes] >> (8 - remainingBits)) == 0;
    }
}
