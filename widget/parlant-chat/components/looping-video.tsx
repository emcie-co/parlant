import { useEffect, useRef } from "react";

export interface LoopingVideoProps {
  /** An H.264 MP4 with its index at the start, so it streams. Not a WebM: iPhone Safari
   *  reports it can play one, then stalls on it. */
  src: string;
  /** The first frame, shown while it loads and in place of it with reduced motion. */
  poster?: string;
  className?: string;
}

/** A muted decorative loop. It plays only while on screen, and with reduced motion it stays
 *  on the poster. */
export function LoopingVideo({ src, poster, className }: LoopingVideoProps) {
  const ref = useRef<HTMLVideoElement>(null);
  useEffect(() => {
    const el = ref.current;
    // Without IntersectionObserver (old browsers, tests) the video stays on its poster.
    if (!el || typeof IntersectionObserver === "undefined") return;
    if (window.matchMedia?.("(prefers-reduced-motion: reduce)").matches) return;
    const observer = new IntersectionObserver(([entry]) => {
      if (entry.isIntersecting) el.play().catch(() => {});
      else el.pause();
    });
    observer.observe(el);
    return () => observer.disconnect();
  }, []);

  return (
    <video
      ref={ref}
      src={src}
      className={className}
      poster={poster}
      muted
      loop
      playsInline
      preload="metadata"
      // No picture-in-picture or cast buttons on hover: it's decoration, not a player.
      disablePictureInPicture
      disableRemotePlayback
      aria-hidden
    />
  );
}
