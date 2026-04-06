import { useEffect, useRef, useState, ReactNode, ReactElement } from 'react';

interface VirtualScrollContainerProps {
  children: ReactNode[];
  height?: string;
  className?: string;
}

const VirtualScroll = ({ children, height, className }: VirtualScrollContainerProps): ReactElement => {
  const containerRef = useRef<HTMLDivElement>(null);
  const [visibleItems, setVisibleItems] = useState<number[]>([]);

  const observer = useRef<IntersectionObserver | null>(null);

  useEffect(() => {
    observer.current = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        const index = parseInt(entry.target.getAttribute('data-index') || '', 10);
        if (!isNaN(index)) {
          if (entry.isIntersecting) {
            setVisibleItems((prev) => [...new Set([...prev, index])]);
          } else {
            setVisibleItems((prev) => prev.filter((id) => id !== index));
          }
        }
      });
    });

    const elements = containerRef.current?.children;
    if (elements) {
      Array.from(elements).forEach((element, index) => {
        element.setAttribute('data-index', index.toString());
        observer.current?.observe(element);
      });
    }
    return () => observer.current?.disconnect();
  }, [children]);

  return (
    <div className={'scroll-container ' + className} ref={containerRef}>
      {children.map((child, index) => (
        <div
          key={index}
          data-index={index}
          className={'item' + (visibleItems.includes(index) ? '' : ` h-[${height ?? '1px'}]`)}>
          {visibleItems.includes(index) ? child : ''}
        </div>
      ))}
    </div>
  );
};

export default VirtualScroll;
