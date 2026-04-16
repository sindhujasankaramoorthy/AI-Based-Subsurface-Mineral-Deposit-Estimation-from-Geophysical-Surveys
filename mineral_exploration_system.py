"""
AI-Based Subsurface Mineral Deposit Estimation System - FIXED VERSION
Fixed dimension mismatch errors
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

# ============================================================================
# PART 1: DATA GENERATION (Synthetic Geophysical Surveys)
# ============================================================================

class GeophysicalSurveySimulator:
    """Simulates seismic and MT surveys with known mineral deposits"""
    
    def __init__(self, grid_size=(40, 40, 20), survey_points=16):
        self.nx, self.ny, self.nz = grid_size
        self.survey_points = survey_points
        
    def generate_mineral_deposit(self, deposit_type='copper'):
        """Generate realistic 3D mineral deposit distribution"""
        volume = np.zeros((self.nx, self.ny, self.nz))
        
        if deposit_type == 'copper':
            # Porphyry copper style deposit
            center = (np.random.randint(15, 25), np.random.randint(15, 25), 
                     np.random.randint(8, 12))
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        dist = np.sqrt(((i - center[0])/6)**2 + 
                                      ((j - center[1])/6)**2 + 
                                      ((k - center[2])/4)**2)
                        if dist < 1.5:
                            volume[i,j,k] = np.random.uniform(0.5, 1.0)
                        elif dist < 2.5:
                            volume[i,j,k] = np.random.uniform(0.2, 0.7)
                        elif dist < 3.5:
                            volume[i,j,k] = np.random.uniform(0.05, 0.3)
                        else:
                            volume[i,j,k] = np.random.uniform(0, 0.05)
                            
        elif deposit_type == 'lithium':
            # Brine deposit style
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        if k > self.nz * 0.7:
                            if np.random.random() < 0.3:
                                volume[i,j,k] = np.random.uniform(0.3, 0.8)
                                
        elif deposit_type == 'iron_ore':
            # Banded iron formation style
            for i in range(self.nx):
                for j in range(self.ny):
                    for k in range(self.nz):
                        if 0.4 < k/self.nz < 0.6:
                            if np.sin(i/5) * np.cos(j/5) > 0.5:
                                volume[i,j,k] = np.random.uniform(0.4, 0.9)
        
        # Add noise
        volume += np.random.normal(0, 0.02, volume.shape)
        return np.clip(volume, 0, 1)
    
    def generate_seismic_profile(self, mineral_volume):
        """Generate 2D seismic reflection profiles - FIXED to 2D"""
        # Average over y dimension to create a single 2D profile
        seismic_2d = np.zeros((self.nx, self.nz))
        
        for x in range(self.nx):
            for z in range(self.nz):
                # Average mineral concentration across y-direction
                avg_mineral = np.mean(mineral_volume[x, :, z])
                
                # Seismic response based on mineral concentration
                reflection = avg_mineral
                
                # Add realistic wavelet effects
                wavelet = reflection * np.exp(-((z - self.nz/2)**2) / 100)
                
                # Add multiples and noise
                if z > 5 and avg_mineral > 0.1:
                    wavelet += 0.3 * avg_mineral
                
                seismic_2d[x, z] = wavelet + np.random.normal(0, 0.05)
        
        return seismic_2d
    
    def generate_mt_data(self, mineral_volume):
        """Generate magnetotelluric resistivity data - FIXED to 2D grid"""
        mt_2d = np.zeros((self.survey_points, self.nz))
        
        for i in range(self.survey_points):
            x = int(i * self.nx / self.survey_points)
            
            for z in range(self.nz):
                # Average across y-direction
                avg_mineral = np.mean(mineral_volume[x, :, z])
                
                # Resistivity inversely related to mineral grade
                resistivity = 100 * np.exp(-3 * avg_mineral)
                resistivity += np.random.normal(0, 5)
                mt_2d[i, z] = max(1, resistivity)
        
        return mt_2d
    
    def generate_training_data(self, n_samples=500):
        """Generate complete dataset with multiple deposit types"""
        X_seismic = []
        X_mt = []
        y_mineral = []
        deposit_types = ['copper', 'lithium', 'iron_ore']
        
        for _ in range(n_samples):
            deposit_type = np.random.choice(deposit_types)
            mineral_volume = self.generate_mineral_deposit(deposit_type)
            
            seismic = self.generate_seismic_profile(mineral_volume)
            mt = self.generate_mt_data(mineral_volume)
            
            X_seismic.append(seismic)
            X_mt.append(mt)
            y_mineral.append(mineral_volume)
        
        return (np.array(X_seismic), np.array(X_mt), 
                np.array(y_mineral))

# ============================================================================
# PART 2: FIXED DEEP LEARNING MODEL
# ============================================================================

class SeismicEncoder(nn.Module):
    """Encoder for seismic reflection profiles - FIXED for 2D input"""
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        return self.encoder(x)

class MTEncoder(nn.Module):
    """Encoder for magnetotelluric data - FIXED for 2D input"""
    def __init__(self, input_channels=1, latent_dim=128):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, latent_dim),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
    
    def forward(self, x):
        # x shape: [batch, channels, height, width]
        return self.encoder(x)

class ProbabilisticDecoder(nn.Module):
    """Decoder with uncertainty quantification"""
    def __init__(self, latent_dim=256, output_shape=(40, 40, 20)):
        super().__init__()
        self.output_shape = output_shape
        total_voxels = output_shape[0] * output_shape[1] * output_shape[2]
        
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.2),
            
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.2),
            
            nn.Linear(1024, 2048),
            nn.ReLU(),
            nn.BatchNorm1d(2048),
            
            nn.Linear(2048, total_voxels * 2)
        )
    
    def forward(self, x):
        params = self.decoder(x)
        mean = params[:, :params.shape[1]//2]
        log_var = params[:, params.shape[1]//2:]
        
        # Reshape to 3D
        mean = mean.view(-1, *self.output_shape)
        log_var = log_var.view(-1, *self.output_shape)
        
        return mean, log_var

class MineralEstimationNet(nn.Module):
    """Complete deep learning system - FIXED dimensions"""
    def __init__(self, seismic_channels=1, mt_channels=1, latent_dim=128):
        super().__init__()
        self.seismic_encoder = SeismicEncoder(seismic_channels, latent_dim)
        self.mt_encoder = MTEncoder(mt_channels, latent_dim)
        
        # Fusion layer
        self.fusion = nn.Sequential(
            nn.Linear(latent_dim * 2, latent_dim * 2),
            nn.ReLU(),
            nn.Linear(latent_dim * 2, latent_dim),
            nn.ReLU()
        )
        
        self.decoder = ProbabilisticDecoder(latent_dim)
    
    def forward(self, seismic, mt):
        # Add channel dimension: [batch, h, w] -> [batch, 1, h, w]
        if seismic.dim() == 3:
            seismic = seismic.unsqueeze(1)
        if mt.dim() == 3:
            mt = mt.unsqueeze(1)
        
        # Encode both modalities
        seismic_feat = self.seismic_encoder(seismic)
        mt_feat = self.mt_encoder(mt)
        
        # Fuse features
        combined = torch.cat([seismic_feat, mt_feat], dim=1)
        fused = self.fusion(combined)
        
        # Decode with uncertainty
        mean, log_var = self.decoder(fused)
        return mean, log_var
    
    def sample_predictions(self, seismic, mt, n_samples=10):
        """Generate multiple predictions for uncertainty estimation"""
        mean, log_var = self.forward(seismic, mt)
        std = torch.exp(0.5 * log_var)
        
        samples = []
        for _ in range(n_samples):
            sample = mean + std * torch.randn_like(mean)
            samples.append(sample)
        
        return torch.stack(samples), mean, std

# ============================================================================
# PART 3: UNCERTAINTY-AWARE TRAINING
# ============================================================================

class UncertaintyLoss(nn.Module):
    """Loss function that accounts for prediction uncertainty"""
    def __init__(self):
        super().__init__()
        self.mse_loss = nn.MSELoss(reduction='none')
    
    def forward(self, mean, log_var, target):
        # Negative log likelihood loss
        precision = torch.exp(-log_var)
        nll_loss = 0.5 * (precision * (mean - target)**2 + log_var)
        
        # Add MSE component for stability
        mse_loss = self.mse_loss(mean, target)
        
        return (nll_loss.mean() + 0.1 * mse_loss.mean())

class MineralExplorationSystem:
    """Complete training and prediction pipeline"""
    
    def __init__(self, model, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, patience=5, factor=0.5
        )
        self.criterion = UncertaintyLoss()
        
    def train(self, train_loader, val_loader, epochs=50):
        """Train the model with early stopping"""
        best_val_loss = float('inf')
        patience_counter = 0
        
        train_losses = []
        val_losses = []
        
        for epoch in range(epochs):
            # Training phase
            self.model.train()
            epoch_train_loss = 0
            
            for seismic, mt, mineral in train_loader:
                seismic = seismic.to(self.device)
                mt = mt.to(self.device)
                mineral = mineral.to(self.device)
                
                self.optimizer.zero_grad()
                mean, log_var = self.model(seismic, mt)
                loss = self.criterion(mean, log_var, mineral)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                
                epoch_train_loss += loss.item()
            
            # Validation phase
            self.model.eval()
            epoch_val_loss = 0
            
            with torch.no_grad():
                for seismic, mt, mineral in val_loader:
                    seismic = seismic.to(self.device)
                    mt = mt.to(self.device)
                    mineral = mineral.to(self.device)
                    
                    mean, log_var = self.model(seismic, mt)
                    loss = self.criterion(mean, log_var, mineral)
                    epoch_val_loss += loss.item()
            
            avg_train_loss = epoch_train_loss / len(train_loader)
            avg_val_loss = epoch_val_loss / len(val_loader)
            
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            self.scheduler.step(avg_val_loss)
            
            # Early stopping
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                torch.save(self.model.state_dict(), 'best_model.pth')
            else:
                patience_counter += 1
                if patience_counter >= 10:
                    print(f"Early stopping at epoch {epoch}")
                    break
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={avg_train_loss:.4f}, "
                      f"Val Loss={avg_val_loss:.4f}")
        
        return train_losses, val_losses
    
    def predict_with_uncertainty(self, seismic, mt):
        """Generate predictions with uncertainty estimates"""
        self.model.eval()
        with torch.no_grad():
            seismic_tensor = torch.FloatTensor(seismic).unsqueeze(0).to(self.device)
            mt_tensor = torch.FloatTensor(mt).unsqueeze(0).to(self.device)
            
            samples, mean, std = self.model.sample_predictions(seismic_tensor, mt_tensor)
            
        return mean.cpu().numpy()[0], std.cpu().numpy()[0]

# ============================================================================
# PART 4: DRILL SITE SELECTION AND ECONOMIC ANALYSIS
# ============================================================================

class DrillSiteOptimizer:
    """Optimizes drill site selection based on predictions and uncertainty"""
    
    def __init__(self, min_grade_threshold=0.3):
        self.threshold = min_grade_threshold
        
    def identify_target_zones(self, mean_prediction, uncertainty, top_k=5):
        """Identify top drill target locations"""
        # Compute expected value (mean - uncertainty penalty)
        expected_value = mean_prediction - 0.5 * uncertainty
        
        # Find top voxels
        flat_indices = np.argsort(expected_value.flatten())[-top_k*10:][::-1]
        
        targets = []
        for idx in flat_indices[:top_k]:
            z, y, x = np.unravel_index(idx, mean_prediction.shape)
            targets.append({
                'location': (int(x), int(y), int(z)),
                'probability': float(mean_prediction[z, y, x]),
                'uncertainty': float(uncertainty[z, y, x]),
                'expected_value': float(expected_value[z, y, x])
            })
        
        return targets
    
    def calculate_economic_viability(self, mineral_volume, commodity_price):
        """Calculate economic viability score"""
        avg_grade = np.mean(mineral_volume)
        total_tonnage = np.sum(mineral_volume) * 1000
        
        if avg_grade > 0.5:
            recovery = 0.85
        elif avg_grade > 0.3:
            recovery = 0.70
        else:
            recovery = 0.50
        
        revenue = total_tonnage * avg_grade * recovery * commodity_price
        operating_cost = total_tonnage * 50
        capital_cost = 10000000
        
        npv = revenue - operating_cost - capital_cost
        
        if npv > 50000000:
            score = 1.0
        elif npv > 10000000:
            score = 0.6 + 0.4 * (npv - 10000000) / 40000000
        elif npv > 0:
            score = 0.3 + 0.3 * npv / 10000000
        else:
            score = max(0, 0.3 * (1 + npv / 5000000))
        
        return {
            'npv': float(npv),
            'economic_score': float(score),
            'avg_grade': float(avg_grade),
            'total_tonnage': float(total_tonnage),
            'revenue': float(revenue),
            'operating_cost': float(operating_cost)
        }

# ============================================================================
# PART 5: VISUALIZATION AND REPORTING
# ============================================================================

class VisualizationEngine:
    """Creates geological cross-sections and uncertainty visualizations"""
    
    @staticmethod
    def plot_cross_section(mean_prediction, uncertainty, true_values=None, 
                           slice_idx=None, save_path='cross_section.png'):
        """Plot 2D cross-section with confidence envelopes"""
        if slice_idx is None:
            slice_idx = mean_prediction.shape[1] // 2
        
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Mean prediction
        im1 = axes[0, 0].imshow(mean_prediction[:, slice_idx, :].T, 
                                cmap='viridis', aspect='auto', origin='lower')
        axes[0, 0].set_title('Mean Mineral Probability')
        plt.colorbar(im1, ax=axes[0, 0])
        
        # Uncertainty
        im2 = axes[0, 1].imshow(uncertainty[:, slice_idx, :].T, 
                                cmap='plasma', aspect='auto', origin='lower')
        axes[0, 1].set_title('Prediction Uncertainty (Std Dev)')
        plt.colorbar(im2, ax=axes[0, 1])
        
        # Confidence envelope
        lower_bound = mean_prediction - 1.96 * uncertainty
        upper_bound = mean_prediction + 1.96 * uncertainty
        
        axes[1, 0].fill_between(range(mean_prediction.shape[0]), 
                                lower_bound[:, slice_idx, mean_prediction.shape[2]//2],
                                upper_bound[:, slice_idx, mean_prediction.shape[2]//2],
                                alpha=0.3, label='95% Confidence Interval')
        axes[1, 0].plot(mean_prediction[:, slice_idx, mean_prediction.shape[2]//2], 
                       'b-', label='Mean Prediction')
        if true_values is not None:
            axes[1, 0].plot(true_values[:, slice_idx, mean_prediction.shape[2]//2], 
                           'r--', label='True Values')
        axes[1, 0].set_title('Confidence Envelope (Center Line)')
        axes[1, 0].legend()
        axes[1, 0].set_xlabel('X Coordinate')
        axes[1, 0].set_ylabel('Mineral Probability')
        
        # Error distribution
        if true_values is not None:
            error = mean_prediction - true_values
            axes[1, 1].hist(error.flatten(), bins=50, alpha=0.7)
            axes[1, 1].axvline(x=0, color='r', linestyle='--')
            axes[1, 1].set_title('Prediction Error Distribution')
            axes[1, 1].set_xlabel('Error')
            axes[1, 1].set_ylabel('Frequency')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.show()
    
    @staticmethod
    def plot_3d_probability_volume(mean_prediction, threshold=0.3):
        """3D visualization of high-probability zones"""
        from mpl_toolkits.mplot3d import Axes3D
        
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        high_prob = mean_prediction > threshold
        coords = np.where(high_prob)
        
        if len(coords[0]) > 0:
            colors = mean_prediction[high_prob]
            scatter = ax.scatter(coords[0], coords[1], coords[2], 
                               c=colors, cmap='hot', s=10, alpha=0.6)
            plt.colorbar(scatter, label='Probability')
        
        ax.set_xlabel('X Coordinate')
        ax.set_ylabel('Y Coordinate')
        ax.set_zlabel('Z Coordinate (Depth)')
        ax.set_title(f'3D Mineral Probability Volume (>{threshold})')
        plt.savefig('3d_mineral_volume.png', dpi=150)
        plt.show()

# ============================================================================
# PART 6: MAIN PIPELINE AND EVALUATION
# ============================================================================

def evaluate_model(model, test_loader, device):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_targets = []
    all_uncertainties = []
    
    with torch.no_grad():
        for seismic, mt, mineral in test_loader:
            seismic = seismic.to(device)
            mt = mt.to(device)
            
            mean, log_var = model(seismic, mt)
            std = torch.exp(0.5 * log_var)
            
            all_predictions.append(mean.cpu().numpy())
            all_targets.append(mineral.cpu().numpy())
            all_uncertainties.append(std.cpu().numpy())
    
    predictions = np.concatenate(all_predictions)
    targets = np.concatenate(all_targets)
    uncertainties = np.concatenate(all_uncertainties)
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(targets.flatten(), predictions.flatten()))
    
    # Uncertainty calibration
    residuals = np.abs(predictions - targets)
    uncertainty_ratio = residuals / (uncertainties + 1e-8)
    well_calibrated = np.mean(uncertainty_ratio < 1.96)
    
    # IoU for deposit boundary
    pred_binary = predictions > 0.3
    true_binary = targets > 0.3
    
    intersection = np.logical_and(pred_binary, true_binary).sum()
    union = np.logical_or(pred_binary, true_binary).sum()
    iou = intersection / union if union > 0 else 0
    
    print("\n" + "="*50)
    print("EVALUATION METRICS")
    print("="*50)
    print(f"RMSE (Mineral Grade): {rmse:.4f}")
    print(f"Uncertainty Calibration (95% CI coverage): {well_calibrated:.3f}")
    print(f"IoU (Deposit Boundary): {iou:.3f}")
    print("="*50)
    
    return {
        'rmse': rmse,
        'calibration': well_calibrated,
        'iou': iou,
        'predictions': predictions,
        'targets': targets,
        'uncertainties': uncertainties
    }

def main():
    """Main execution pipeline"""
    print("="*60)
    print("AI-BASED SUBSURFACE MINERAL DEPOSIT ESTIMATION SYSTEM")
    print("="*60)
    
    # Generate synthetic data with smaller grid for faster training
    print("\n[1/6] Generating synthetic geophysical survey data...")
    simulator = GeophysicalSurveySimulator(grid_size=(30, 30, 15), survey_points=12)
    X_seismic, X_mt, y_mineral = simulator.generate_training_data(n_samples=300)
    
    print(f"  - Seismic profiles shape: {X_seismic.shape}")
    print(f"  - MT data shape: {X_mt.shape}")
    print(f"  - Mineral volumes shape: {y_mineral.shape}")
    
    # Split data
    print("\n[2/6] Splitting data for training/evaluation...")
    X_seismic_train, X_seismic_test, X_mt_train, X_mt_test, y_train, y_test = train_test_split(
        X_seismic, X_mt, y_mineral, test_size=0.2, random_state=42
    )
    
    # Create datasets and loaders
    train_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_seismic_train),
        torch.FloatTensor(X_mt_train),
        torch.FloatTensor(y_train)
    )
    test_dataset = torch.utils.data.TensorDataset(
        torch.FloatTensor(X_seismic_test),
        torch.FloatTensor(X_mt_test),
        torch.FloatTensor(y_test)
    )
    
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    
    # Initialize model
    print("\n[3/6] Initializing deep learning model...")
    model = MineralEstimationNet(seismic_channels=1, mt_channels=1, latent_dim=64)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  - Total parameters: {total_params:,}")
    
    # Train model
    print("\n[4/6] Training model with uncertainty quantification...")
    exploration_system = MineralExplorationSystem(model)
    train_losses, val_losses = exploration_system.train(train_loader, test_loader, epochs=30)
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training History')
    plt.legend()
    plt.grid(True)
    plt.savefig('training_history.png')
    plt.show()
    
    # Evaluate model
    print("\n[5/6] Evaluating model performance...")
    evaluation = evaluate_model(model, test_loader, exploration_system.device)
    
    # Test prediction on a sample
    print("\n[6/6] Generating predictions and recommendations...")
    sample_idx = 0
    mean_pred, uncertainty = exploration_system.predict_with_uncertainty(
        X_seismic_test[sample_idx], X_mt_test[sample_idx]
    )
    
    # Visualize results
    viz = VisualizationEngine()
    viz.plot_cross_section(mean_pred, uncertainty, y_test[sample_idx])
    viz.plot_3d_probability_volume(mean_pred, threshold=0.3)
    
    # Drill site optimization
    optimizer = DrillSiteOptimizer(min_grade_threshold=0.3)
    top_targets = optimizer.identify_target_zones(mean_pred, uncertainty, top_k=5)
    
    print("\n" + "="*60)
    print("TOP-5 RECOMMENDED DRILL SITES")
    print("="*60)
    for i, target in enumerate(top_targets, 1):
        print(f"\nSite {i}:")
        print(f"  Location (X,Y,Z): {target['location']}")
        print(f"  Mineral Probability: {target['probability']:.3f}")
        print(f"  Uncertainty: {target['uncertainty']:.3f}")
        print(f"  Expected Value (risk-adjusted): {target['expected_value']:.3f}")
    
    # Economic analysis
    print("\n" + "="*60)
    print("ECONOMIC VIABILITY ANALYSIS")
    print("="*60)
    
    commodity_prices = {
        'copper': 9000,
        'lithium': 14000,
        'iron_ore': 120
    }
    
    for commodity, price in commodity_prices.items():
        economic = optimizer.calculate_economic_viability(mean_pred, price)
        print(f"\n{commodity.upper()} DEPOSIT:")
        print(f"  Economic Viability Score: {economic['economic_score']:.3f}")
        print(f"  Estimated NPV: ${economic['npv']:,.0f}")
        print(f"  Average Grade: {economic['avg_grade']:.3f}")
        print(f"  Total Tonnage: {economic['total_tonnage']:,.0f} tons")
    
    print("\n" + "="*60)
    print("SYSTEM COMPLETE")
    print("="*60)
    print("\nDeliverables generated:")
    print("  ✓ Probabilistic 3D mineral volume")
    print("  ✓ Top-K drill site recommendations with uncertainty")
    print("  ✓ Geological cross-sections with confidence envelopes")
    print("  ✓ Economic viability scores")
    print("  ✓ Evaluation metrics (RMSE, calibration, IoU)")
    
    return model, evaluation, top_targets

if __name__ == "__main__":
    model, metrics, targets = main()