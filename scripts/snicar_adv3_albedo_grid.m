% Writes data/snicar_adv3_albedo.csv: broadband albedo of optically semi-infinite snow from
% SNICAR-ADv3 (Flanner et al. 2021, GMD 14, 7673-7704), over the grain radii, black carbon and
% dust loadings that `background_black_carbon` converts to a Gardner & Sharp (2010) loading.
%
% Code: github.com/mflanner/SNICARv3 at commit a122b20782f68c8dad62c1ea8037fbfda7a488a8, with
% dir_op_root in snicarAD_v3.m set to the unpacked optics library snicar_v3_optics.tar.gz
% (snicar_480band). Run from a directory holding that snicarAD_v3.m:
%
%     matlab -batch "addpath(pwd); snicar_adv3_albedo_grid('data/snicar_adv3_albedo.csv')"
%
% Settings: spheres; uncoated (externally mixed) BC; Saharan dust (dust_type 1, central hematite);
% merged Picard et al. (2016) / Warren and Brandt (2008) ice refractive index; sub-Arctic summer
% atmosphere; direct beam at a 49.5 degree solar zenith angle, or diffuse (SNICAR's cloudy-sky
% spectrum). Dust mass is split over SNICAR's five radius bins in proportion to the lognormal
% (mass median diameter 3.5 um, sigma_g = 2) its dust optics are computed from.
function snicar_adv3_albedo_grid(path)
radii = [50 100 200 300 500];
bcs   = [0 1 2 5 10 20 30 50 100];
dusts = [0 0.5 1 2 5 10 20 30 40 50];
edges = [0.05 0.5 1.25 2.5 5.0 50];
cdf = 0.5*(1 + erf(log(edges/1.75)/(log(2)*sqrt(2))));
fr = diff(cdf); fr = fr/sum(fr);
names = {'diffuse', 'direct'};
fid = fopen(path, 'w');
fprintf(fid, 'illumination,radius_um,bc_ng_g,dust_ug_g,albedo\n');
for illum = [1 0]
  for r = radii
    for bc = bcs
      for du = dusts
        ia = struct();
        ia.direct_beam = illum; ia.coszen = cosd(49.5);
        ia.dz = 100; ia.rho_snw = 200; ia.rds_snw = r; ia.ice_ri = 3;
        ia.sno_shp = 1; ia.sno_fs = 0; ia.sno_ar = 0;
        ia.dust_type = 1; ia.ash_type = 1;
        ia.mss_cnc_sot1 = bc; ia.mss_cnc_sot2 = 0;
        ia.mss_cnc_brc1 = 0; ia.mss_cnc_brc2 = 0;
        ia.mss_cnc_dst1 = du*fr(1); ia.mss_cnc_dst2 = du*fr(2); ia.mss_cnc_dst3 = du*fr(3);
        ia.mss_cnc_dst4 = du*fr(4); ia.mss_cnc_dst5 = du*fr(5);
        ia.mss_cnc_ash1 = 0; ia.mss_cnc_ash2 = 0; ia.mss_cnc_ash3 = 0;
        ia.mss_cnc_ash4 = 0; ia.mss_cnc_ash5 = 0;
        ia.cell_nbr_conc = 0; ia.alg_rds = 10; ia.dcmf_pig_chla = 0.015;
        ia.dcmf_pig_chlb = 0.005; ia.dcmf_pig_cara = 0.05; ia.dcmf_pig_carb = 0;
        ia.R_sfc_all_wvl = 0.25; ia.atm = 4; ia.flx_dwn_bb = 1.0;
        d = snicarAD_v3(ia);
        fprintf(fid, '%s,%g,%g,%g,%.8f\n', names{illum + 1}, r, bc, du, d.alb_slr);
      end
    end
  end
end
fclose(fid);
end
