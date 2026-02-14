import logo from '../../assets/credresolve_logo.jpeg';

export function Header() {
  return (
    <header className="w-full py-4 px-4 flex justify-center">
      <img
        src={logo}
        alt="CredResolve logo"
        className="h-12 w-auto object-contain sm:h-14"
      />
    </header>
  );
}
