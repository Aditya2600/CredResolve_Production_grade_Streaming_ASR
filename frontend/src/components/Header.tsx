import logo from '../../assets/credresolve_logo.jpeg';

export function Header() {
  return (
    <header className="w-full py-6 px-4 flex justify-center">
      <img
        src={logo}
        alt="CredResolve logo"
        className="h-20 w-auto object-contain sm:h-24 drop-shadow-md transition-transform hover:scale-105"
      />
    </header>
  );
}
